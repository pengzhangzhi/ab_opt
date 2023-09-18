import torch
import torch.nn as nn

from src.modules.common.geometry import construct_3d_basis
from src.modules.common.so3 import rotation_to_so3vec
from src.modules.encoders.residue import ResidueEmbedding
from src.modules.encoders.pair import PairEmbedding
from src.modules.diffusion.dpm_full import FullDPM
from src.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom, NUM_BB_ATOMS
from ._base import register_model


resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


@register_model('diffab')
class DiffusionAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            **cfg.diffusion,
            num_bins=cfg.num_bins,
            dist_min=cfg.dist_min,
            dist_max=cfg.dist_max
        )

    def encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['generate_flag']     # Context means ``not generated''
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
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
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p
    
    def forward(self, batch):
        mask_generate = batch['generate_flag']
        # if self.cfg.get("bb_noise",False) \
        #     and self.cfg.get('train_sequence', True) is True:
        #     # add noise to backbone coordinate 
        #     bb_coords = batch['pos_heavyatom'][..., :NUM_BB_ATOMS,:]
        #     noise = self.cfg.bb_noise * torch.randn_like(bb_coords)
        #     batch['pos_heavyatom'][..., :NUM_BB_ATOMS,:] = bb_coords + noise
            
        if self.cfg.get('mask_ratio_min', False):
            random_mask = generate_random_mask_from(mask_generate, self.cfg.mask_ratio_min, self.cfg.mask_ratio_max)
            mask_generate = torch.logical_and(mask_generate, random_mask)
            batch['generate_flag'] = mask_generate
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        loss_dict = self.diffusion(
            v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence  = self.cfg.get('train_sequence', True),
        )
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
            "contig": "",
        }
    ):
        mask_generate = batch['generate_flag']
        if sample_opt.get('sample_sequence', False)\
            and sample_opt['contig'] != '':
            mask = generate_mask_from_str(sample_opt['contig'], mask_generate)
            mask_generate = torch.logical_and(mask_generate, mask)
            batch['generate_flag'] = mask_generate
           
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        traj = self.diffusion.sample(v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, **sample_opt)
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


def generate_random_mask_from(tensor, mask_ratio_min, mask_ratio_max):
    """
    generate mask tensor with ratio between min and max.

    Args:
        mask_ratio_min (float): 
        mask_ratio_max (float): 
    
    """
    mask_ratio = float(torch.empty(1).uniform_(mask_ratio_min, mask_ratio_max))

    probs = torch.zeros_like(tensor.float()).fill_(mask_ratio)
    mask = torch.bernoulli(probs)
    mask = mask.bool()
    return mask



def generate_mask_from_str(str_input, tensor):
    """
    Generate a mask tensor from a string input. 
    The string input should be of the form 'start-end' where start and end are integers.
    
    Args:
        str_input (str): The string input, of the form 'start-end'
        tensor (torch.Tensor): template tensor to generate the mask from
    Returns:
        mask (torch.Tensor): The generated mask tensor, with the same shape as the input tensor. 
                                mask[..., start-1:end] = True, and other entries are False.
    """
    # Parse the string input to extract the start and end indices
    start, end = str_input.split('-')
    start, end = int(start), int(end)
    
    # Generate a mask tensor with all entries set to False
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    
    # Set the entries from start to end to True
    mask[..., start-1:end] = True
    
    return mask