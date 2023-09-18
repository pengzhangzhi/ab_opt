import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from src.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from src.modules.common.plddt import PerResidueLDDTCaPredictor, compute_plddt, lddt_loss
from src.modules.common.prmsd import PerResidueRMSDCaPredictor, pRMSDCa
from src.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from src.modules.encoders.ga import GAEncoder
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, no_bins, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1) 
        )
        self.prmsd_predictor = PerResidueRMSDCaPredictor(
            no_bins, res_feat_dim+3, res_feat_dim,
        )
        # self.plddt_predictor = PerResidueLDDTCaPredictor(
        #     no_bins, res_feat_dim, res_feat_dim,
        # )

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            beta:   (N,).
            mask_generate:    (N, L).
            mask_res:       (N, L).
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t) # (N, L, 3, 3)

        # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)) # [Important] Incorporate sequence at the current step.
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        # Position changes
        eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        # New orientation
        eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        # New sequence categorical distributions
        c_denoised = self.eps_seq_net(in_feat)  # Already softmax-ed, (N, L, 20)
        prmsd_logits = self.prmsd_predictor(in_feat)  # (N, L, no_bins)
        prmsd_logits = prmsd_logits.mean(dim=1)  # (N, no_bins)
        # prmsd_logits = self.plddt_predictor(res_feat)  # (N, L, no_bins)
        return v_next, R_next, eps_pos, c_denoised, prmsd_logits


class FullDPM(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
        obj = 'pred_noise',
        num_bins=20,
        dist_min=0.5,
        dist_max=19.5
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt, no_bins=num_bins)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))
        self.obj = obj
        assert self.obj in ['pred_x0', 'pred_noise']
        self.prmsd = pRMSDCa(num_bins, dist_min=dist_min, dist_max=dist_max)
    
    
    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, denoise_structure, denoise_sequence, t=None):
        N, L = res_feat.shape[:2]
        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            # Add noise to rotation
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            # Add noise to positions
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            # Add noise to sequence
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
        v_pred, R_pred, p_pred, c_denoised, prmsd_logits = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)
        if self.obj == 'pred_x0':
            p_true = p_0
            pred_p0 = p_pred
        elif self.obj == 'pred_noise':
            p_true = p_noisy
            pred_p0 = self.trans_pos.pred_start_from_noise(p_0, p_pred,mask_generate, t)
        loss_dict = {}
        
        # PRMSD loss
        # per_rmsd = pRMSDCa.calc_per_rmsd(self._unnormalize_position(pred_p0), self._unnormalize_position(p_0))
        rmsd = pRMSDCa.calc_rmsd(self._unnormalize_position(pred_p0), self._unnormalize_position(p_0), mask_generate)
        prmsd_loss = self.prmsd(prmsd_logits, rmsd.detach(), mask_generate[:, 0])
        # prmsd_loss = self.prmsd(prmsd_logits, per_rmsd.detach(), mask_generate)
        loss_dict['prmsd'] = prmsd_loss
        
        # # pLDDT loss
        # plddt_loss = lddt_loss(
        #     prmsd_logits, 
        #     self._unnormalize_position(pred_p0), self._unnormalize_position(p_0), 
        #     mask_generate.unsqueeze(-1)
        # )
        # loss_dict['plddt'] = plddt_loss
        
        # distance loss
        if self.obj == 'pred_x0':
            dist_loss = calc_dist_loss(p_pred, p_true, mask_generate, mask_res)
            loss_dict['dist'] = dist_loss
        
        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(p_pred, p_true, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred, 
            target=post_true, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        v, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False, **kwargs
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: [v_init, self._unnormalize_position(p_init), s_init, torch.zeros_like(s_init), torch.ones_like(s_init)]}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t,_,__ = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, p_pred, c_denoised, prmsd_logits = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)
            prmsd_score = self.prmsd.compute_prmsd(prmsd_logits)
            # plddt_score = compute_plddt(prmsd_logits)
            if self.obj == 'pred_x0':
                eps_p = self.trans_pos.pred_noise_from_start(p_t, p_pred, mask_generate, t_tensor)
            elif self.obj == 'pred_noise':
                eps_p = p_pred 
            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            logits, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)
            perplexity = calc_perplexity(logits,mask_generate)
            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = [v_next, self._unnormalize_position(p_next), s_next]+[prmsd_score.cpu(), perplexity.cpu()]
            traj[t] = list(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, 
        opt_step: int,
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init, torch.zeros_like(s_init), torch.ones_like(s_init))}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t,_,__ = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised,prmsd_logits = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)
            prmsd_score = self.prmsd.compute_prmsd(prmsd_logits)
            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            logits, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)
            perplexity = calc_perplexity(logits)
            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next, prmsd_score, perplexity)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

def calc_dist_loss(p_pred, p_true, mask_generate, mask_res):
    pred_dist_map = torch.cdist(p_pred, p_pred)   # (N, L, L)
    true_dist_map = torch.cdist(p_true, p_true)   # (N, L, L)
    # repeat mask_generate to match the shape of distance map
    mask_map = mask_res[:, :, None] & mask_res[:, None, :]  # (N, L, L)
    mask_generate_expand = mask_generate[:, :, None].expand_as(pred_dist_map) & mask_map # (N, L, L)
    pred_dist = torch.masked_select(pred_dist_map, mask_generate_expand)
    true_dist = torch.masked_select(true_dist_map, mask_generate_expand)
    loss = nn.SmoothL1Loss(reduction='none')(pred_dist, true_dist).mean()
    return loss

def calc_perplexity(logits, mask_generate=None):
    """
    calculate the perplexity of a sequence of logits

    Args:
        logits (torch.Tensor): (*, L, V)
        mask_generate (torch.Tensor): (*, L)

    Returns:
        perplexity (torch.Tensor): (*, )
    """
    if mask_generate is None:
        mask_generate = torch.ones_like(logits[..., 0], dtype=torch.bool)
    max_probs = F.softmax(logits, dim=-1).max(dim=-1)[0]
    # log_probs = F.log_softmax(logits, dim=-1)
    # max_probs = -1 * torch.max(log_probs, dim=-1)[0]
    max_probs = max_probs * mask_generate.float()
    perplexity = max_probs.sum(dim=-1)
    perplexity = perplexity / mask_generate.float().sum(dim=-1)
    return perplexity