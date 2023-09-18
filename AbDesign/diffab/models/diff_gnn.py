import functools
from random import random
from torch_scatter import scatter_sum
import torch
import torch.nn as nn
from diffab.modules.MCAttGNN.mc_att_model import ProteinFeature
from diffab.modules.MCAttGNN.mc_egnn import MCAttEGNN, SeqGNN
from diffab.modules.diffusion.transition import PositionTransition, RotationTransition
from diffab.utils.protein.constants import num_aa_types,AA
from diffab.modules.common.geometry import construct_3d_basis, local_to_global
from diffab.modules.common.so3 import random_uniform_so3, rotation_to_so3vec, so3vec_to_rotation
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.diffusion.dpm_full import rotation_matrix_cosine_loss
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom,backbone_atom_coordinates_tensor
from ._base import register_model
import torch.nn.functional as F
from tqdm import tqdm
from diffab.utils.misc import pair2edge,batchfy,clash_loss
resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


@register_model('diff_gnn')
class DiffusionGNN(nn.Module):

    def __init__(
        self,
        res_feat_dim=128,
        edge_feat_dim=1,
        hidden_size=128,
        num_steps=100,
        num_atoms=3,
        n_layers=6,
        dropout=0.1, 
        dense=False,
        self_condition=False,
        hotspot=True,
        objective='pred_x0', 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        position_mean=[0.0],
        position_scale=[10.0],
        train_structure=True,
        train_sequence=True,
        *args, **kwargs
        ):
        super().__init__()

        self.num_atoms = 3
        self.num_steps = num_steps
        self.self_condition = self_condition
        assert objective in ['pred_x0', 'pred_noise']
        self.objective = objective
        self.protein_edge_constructor = ProteinFeature()
        self.embed = GraphEmbedding(
            res_feat_dim, edge_feat_dim, self.num_atoms,
            hotspot=hotspot,remove_structure=train_structure,
            remove_sequence=train_sequence
            )

        self.gnn = MCAttEGNN(
            res_feat_dim, hidden_size, hidden_size,
            num_atoms, edge_feat_dim, n_layers=n_layers,
            residual=True, dropout=dropout, dense=dense,
            self_condition=self_condition
        )
        self.seqgnn = SeqGNN(
            hidden_size, hidden_size, num_aa_types,
            num_atoms, 0
            )

        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.register_buffer('position_mean', torch.FloatTensor(position_mean))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale))
        self.register_buffer('_dummy', torch.empty([0, ]))
    
    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p
        
    def init_mask(self, aa, coord, generate_range):
        """
        mask the generated residues of aa and coord.
        construct pseudo coordinates of `to-be-generated` residues by interpolating them from the neighboring residues.
        this prevents from leaking the information of `to-be-generated` residues
        Args:
            aa (torch.Tensor): [nb_all_res,]
            coord (torch.Tensor): [nb_all_res, nb_atoms, 3]
            generate_range (torch.Tensor): [bs, 2]
        """
        coord, aa, cmask = coord.clone(), aa.clone(), torch.zeros_like(coord)
        n_channel, n_dim = coord.shape[1:]
        for start, end in generate_range:
            aa[start:end + 1] = AA.UNK
            l_coord, r_coord = coord[start - 1], coord[end + 1]  # [n_channel, 3]
            n_span = end - start + 2
            coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, n_channel, n_dim)  # [n_mask, n_channel, 3]
            coord_offsets = torch.cumsum(coord_offsets, dim=0)
            mask_coords = l_coord + coord_offsets / n_span
            coord[start:end + 1] = mask_coords
            cmask[start:end + 1, ...] = 1
        return coord, aa, cmask
    
    def forward(self, batch):
        graph_data = batch['graph']
        coord = graph_data['coord'][...,:self.num_atoms,:]
        aa = graph_data['aa']
        batch_id = graph_data['batch_id']
        masked_coord, masked_aa, cmask = self.init_mask(aa, coord, graph_data['genearte_flag_range'])
        mask_gen = cmask[:, 0, 0].bool()  # [n_all_node]
        ctx_edges, inter_edges, ctx_edge_feats =self.protein_edge_constructor(
            masked_coord, batch_id, graph_data['fragment_id']
        )
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)
        h0, ctx_edge_feats, iter_edge_feats = self.embed(batch, ctx_edges,inter_edges, lengths)
        coord = self._normalize_position(coord)
        R_0 = construct_3d_basis(
            coord[...,BBHeavyAtom.CA,:],
            coord[...,BBHeavyAtom.C,:],
            coord[...,BBHeavyAtom.N,:],
        )
        v_0 = rotation_to_so3vec(R_0)
        p_0 = coord[...,BBHeavyAtom.CA,:]


        bs = lengths.shape[0]
        t = torch.randint(0, self.num_steps, (bs,), dtype=torch.long, device=self._dummy.device)
        # [nb_all_res, ]
        expanded_t = torch.cat([time.repeat(l) for time,l in zip(t,lengths)])

        v_noisy, _ = self.trans_rot.add_noise(v_0, mask_gen, expanded_t)
        p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_gen,  expanded_t)

        noised_coord = reconstruct_noised_coord(coord, p_noisy, v_noisy, mask_gen)
        noise_condition = self.trans_pos.var_sched.betas[expanded_t]
        h_self_cond = None
        hotspot_label = batch['hotspot_label'][batch['mask']] # [nb_all_res,]
        # original hotspot_label contains 2 for hotspot and 1 for non-hotspot. 
        # cast to 1 for hotspot and 0 for non-hotspot.
        hotspot_label = torch.where(hotspot_label == 2, torch.ones_like(hotspot_label), torch.zeros_like(hotspot_label))
        hydropathy = batch['hydropathy'][batch['mask']]
        charge = batch['charge'][batch['mask']]
        h, z = self.gnn(
            h0, noised_coord,
            ctx_edges, inter_edges,
            noise_condition,class_condition=hotspot_label, 
            hydropathy=hydropathy, charge=charge,
            h_self_cond=h_self_cond,
            ctx_edge_attr=ctx_edge_feats,
            att_edge_attr=None,
        )
        aa_logits = self.seqgnn(
            h, inter_edges,z,None,
            hydropathy=hydropathy, charge=charge,
            )
        R_pred = construct_3d_basis(
            z[...,BBHeavyAtom.CA,:],
            z[...,BBHeavyAtom.C,:],
            z[...,BBHeavyAtom.N,:],
        )
        p_pred = z[...,BBHeavyAtom.CA,:]
        if self.objective == 'pred_x0':
            p0_pred = p_pred
            eps_p_pred = self.trans_pos.pred_noise_from_start(p_noisy, p0_pred, mask_gen, expanded_t)
            p_true = p_0
        elif self.objective == 'pred_noise':
            NotImplementedError()
        loss_dict = {}
        loss_rot = rotation_matrix_cosine_loss(R_pred[mask_gen], R_0[mask_gen]).mean()
        loss_dict['rot'] = loss_rot

        loss_pos = F.mse_loss(p_pred[mask_gen], p_true[mask_gen], reduction='none').sum(dim=-1).mean()
        loss_dict['Ca-pos'] = loss_pos 
        
        bb_loss = F.mse_loss(z[mask_gen], coord[mask_gen], reduction='none').sum(dim=-1).mean()
        loss_dict['bb-pos'] = bb_loss
        # seq loss
        loss_seq = F.cross_entropy(aa_logits[mask_gen], aa[mask_gen])
        loss_dict['seq'] = loss_seq 
        accuracy = torch.sum(aa_logits[mask_gen].softmax(dim=-1).argmax(dim=-1) == aa[mask_gen]) / aa[mask_gen].size(0)
        loss_dict['accuracy'] = accuracy.item() 
        
        
        # clash loss
        # Experiment shows that
        # 1) clash loss converges very soon. 
        # 2) result in RMSD increase by 0.5A in test set. 
        # 3) predicted CDR overly extend to the epitope region 
        # batch_p0_pred = batchfy(p0_pred,lengths)
        # loss_clash = clash_loss(batch_p0_pred,batch['mask'],batch['chain_nb'],3.0078/self.position_scale)
        # loss_dict['cdr2epitope_clash'] = loss_clash
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        graph_data = batch['graph']
        coord = graph_data['coord'][...,:self.num_atoms,:]
        aa = graph_data['aa']
        batch_id = graph_data['batch_id']
        masked_coord, masked_aa, cmask = self.init_mask(aa, coord, graph_data['genearte_flag_range'])
        mask_gen = cmask[:, 0, 0].bool()  # [n_all_node]
        ctx_edges, inter_edges, ctx_edge_feats =self.protein_edge_constructor(
            masked_coord, batch_id, graph_data['fragment_id']
        )
        coord = self._normalize_position(coord)
        R_0 = construct_3d_basis(
            coord[...,BBHeavyAtom.CA,:],
            coord[...,BBHeavyAtom.C,:],
            coord[...,BBHeavyAtom.N,:],
        )
        v_0 = rotation_to_so3vec(R_0)
        p_0 = coord[...,BBHeavyAtom.CA,:]

        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)
        bs = lengths.shape[0]
        nb_all_res = v_0.shape[0]
        h0, ctx_edge_feats, iter_edge_feats = self.embed(batch, ctx_edges,inter_edges, lengths)
        
        if sample_opt['sample_structure']:
            v_rand = random_uniform_so3([nb_all_res], device=self._dummy.device)
            p_rand = torch.randn_like(p_0) # make_interpolate_positions(p, mask_generate) # 
            v_init = torch.where(mask_gen[..., None].expand_as(v_0), v_rand, v_0)
            p_init = torch.where(mask_gen[..., None].expand_as(p_0), p_rand, p_0)
        else:
            v_init, p_init = v_0, p_0
        s_init = aa
        
        # [nb_all_res,]
        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        start = None
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            t_tensor = torch.full((bs,), fill_value=t, dtype=torch.long, device=self._dummy.device)
            # [nb_all_res, ]
            expanded_t = torch.cat([time.repeat(l) for time,l in zip(t_tensor,lengths)])
            noise_condition = self.trans_pos.var_sched.betas[expanded_t]
         
            noised_coord = reconstruct_noised_coord(coord, p_t, v_t, mask_gen)
            self_cond = start if self.self_condition else None
            hotspot_label = batch['hotspot_label'][batch['mask']] # [nb_all_res,]
            # original hotspot_label contains 2 for hotspot and 1 for non-hotspot. 
            # cast to 1 for hotspot and 0 for non-hotspot.
            hotspot_label = torch.where(hotspot_label == 2, torch.ones_like(hotspot_label), torch.zeros_like(hotspot_label))
            hydropathy = batch['hydropathy'][batch['mask']] 
            charge = batch['charge'][batch['mask']] 
            h, z = self.gnn(
                h0, noised_coord,
                ctx_edges, inter_edges,
                noise_condition,class_condition=hotspot_label, 
                hydropathy=hydropathy, charge=charge,
                h_self_cond=self_cond,
                ctx_edge_attr=ctx_edge_feats,
                att_edge_attr=None,
            )
            aa_logits = self.seqgnn(
            h, inter_edges,z,None,
            hydropathy=hydropathy, charge=charge,
            )
            R_pred = construct_3d_basis(
                z[...,BBHeavyAtom.CA,:],
                z[...,BBHeavyAtom.C,:],
                z[...,BBHeavyAtom.N,:],
            )
            v_next = rotation_to_so3vec(R_pred)
            p_pred = z[...,BBHeavyAtom.CA,:]
            if self.objective == 'pred_x0':
                p0_pred = p_pred
                eps_p_pred = self.trans_pos.pred_noise_from_start(p_t, p0_pred, mask_gen, expanded_t)
            elif self.objective == 'pred_noise':
                NotImplementedError()
            
            v_next = self.trans_rot.denoise(v_t, v_next, mask_gen, expanded_t)
            # p_next = self.trans_pos.denoise(p_t, eps_p, mask_gen, expanded_t)
            p_next = self.trans_pos.denoise_from_p0(p_t, p0_pred, mask_gen, expanded_t,guidance_kwargs={**batch,"lengths":lengths})
            
            s_next = aa_logits.softmax(dim=-1).argmax(dim=-1)
            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.
        
        for k,v in traj.items():
            # batchfy the trajectory data
            # to be compatible with subsequent code
            traj[k] = tuple(batchfy(x,lengths) for x in v)
        
        return traj
    @torch.no_grad()
    def optimize(
        self, 
        batch, 
        opt_step, 
        optimize_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = optimize_opt.get('sample_structure', True),
            remove_sequence = optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(v_0, p_0, s_0, opt_step, res_feat, pair_feat, mask_generate, mask_res, **optimize_opt)
        return traj

class GraphEmbedding(nn.Module):
    def __init__(self, res_dim, pair_dim, max_num_atoms=3,hotspot=True,remove_structure=True, remove_sequence=True):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = num_aa_types
        self.remove_structure = remove_structure
        self.remove_sequence = remove_sequence
        self.residue_embed = ResidueEmbedding(res_dim, max_num_atoms, hotspot=hotspot)
        self.pair_embed = PairEmbedding(pair_dim, max_num_atoms)

    def forward(self, batch, ctx_edges, iter_edges, lengths):
        # info of generated regions is not leaked out
        mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['generate_flag']     # Context means ``not generated''
        )
        structure_mask = mask if self.remove_structure else None
        sequence_mask = mask if self.remove_sequence else None
        # [B, N, D]
        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
            hotspot =  batch['hotspot_label'],
            # hydropathy = batch['hydropathy'],
            # charge = batch['charge'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )
        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        ) # (B, L, L, D)

        pad_mask = batch['mask'] # (B, N)
        node_feat = res_feat[pad_mask]
        ctx_edge_feats = pair2edge(ctx_edges, lengths, pair_feat)
        iter_edge_feats = pair2edge(iter_edges, lengths, pair_feat)
        return node_feat, ctx_edge_feats, iter_edge_feats

def reconstruct_noised_coord(coord, p_noisy, v_noisy, mask_gen):
    """
    Args:
        coord: (N, num_atoms, 3)
        p_noisy: (N, num_atoms, 3)
        v_noisy: (N, num_atoms, 3)
        mask_gen: (N, ) bool, true denotes the residue to be generated
    """
    coord = coord.clone()
    num_reconstructed_res = mask_gen.sum()
    device = coord.device
    r_noisy = so3vec_to_rotation(v_noisy[mask_gen])
    bb_coords = backbone_atom_coordinates_tensor.clone().to(device)  # (21, 3, 3)
    # [num_reconstructed_res, 3, 3]
    bb_coords = bb_coords[0].unsqueeze(0).repeat(num_reconstructed_res, 1, 1)
    reconstructed_bb_pos = local_to_global(r_noisy, p_noisy[mask_gen], bb_coords) 
    coord[mask_gen] = reconstructed_bb_pos
    return coord


def pairwise_distance_loss(x_true,x_pred,mask):

    true_dist = torch.cdist(x_true,x_true)
    pred_dist = torch.cdist(x_pred,x_pred)
    mask_dist = torch.cdist(mask,mask)
    loss = torch.mean((true_dist-pred_dist).abs()*mask_dist)
    return loss