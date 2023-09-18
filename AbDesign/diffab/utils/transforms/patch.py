import torch

from ._base import _mask_select_data, register_transform
from ..protein import constants

class CUTOFFVIOLATION(Exception):
    pass

@register_transform('patch_around_anchor')
class PatchAroundAnchor(object):

    def __init__(self, initial_patch_size=128, antigen_size=128,max_nb_hotspots=3,is_train=True, dist_cutoff=None,max_distance=40):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.antigen_size = antigen_size
        self.max_nb_hotspots = max_nb_hotspots
        self.is_train = is_train
        self.dist_cutoff = dist_cutoff
        self.max_distance = max_distance
        
    def _center(self, data, origin):
        origin = origin.reshape(1, 1, 3)
        data['pos_heavyatom'] -= origin # (L, A, 3)
        data['pos_heavyatom'] = data['pos_heavyatom'] * data['mask_heavyatom'][:, :, None]
        data['origin'] = origin.reshape(3)
        return data

    def __call__(self, data):
        anchor_flag = data['anchor_flag']   # (L,)
        anchor_points = data['pos_heavyatom'][anchor_flag, constants.BBHeavyAtom.CA]    # (n_anchors, 3)
        antigen_mask = (data['fragment_type'] == constants.Fragment.Antigen)
        antibody_mask = torch.logical_not(antigen_mask)

        if anchor_flag.sum().item() == 0:
            # Generating full antibody-Fv, no antigen given
            data_patch = _mask_select_data(
                data = data,
                mask = antibody_mask,
            )
            data_patch = self._center(
                data_patch,
                origin = data_patch['pos_heavyatom'][:, constants.BBHeavyAtom.CA].mean(dim=0)
            )
            return data_patch

        pos_alpha = data['pos_heavyatom'][:, constants.BBHeavyAtom.CA]  # (L, 3)
        dist_anchor = torch.cdist(pos_alpha, anchor_points).min(dim=1)[0]    # (L, )
        initial_patch_idx = torch.topk(
            dist_anchor,
            k = min(self.initial_patch_size, dist_anchor.size(0)),
            largest=False,
        )[1]   # (initial_patch_size, )

        dist_anchor_antigen = dist_anchor.masked_fill(
            mask = antibody_mask, # Fill antibody with +inf
            value = float('+inf')
        )   # (L, )
        antigen_dist, antigen_patch_idx = torch.topk(
            dist_anchor_antigen, 
            k = min(self.antigen_size, antigen_mask.sum().item()), 
            largest=False, sorted=True
        )    # (ag_size, )
        
        # label the nb_hotspots closest antigen residues as the hotspots 
        # nb_hotspots = int(torch.randint(0, self.max_nb_hotspots+1, (1, ))) if self.is_train else self.max_nb_hotspots
        nb_hotspots = 1 if self.max_nb_hotspots == 0 else self.max_nb_hotspots
        nb_hotspots = min(nb_hotspots, antigen_mask.sum().item())
        dist_cdr = torch.cdist(
            pos_alpha,
            pos_alpha[data['generate_flag']]
        ).min(dim=1)[0]
        dist_cdr = dist_cdr.masked_fill(
            mask = antibody_mask,
            value = float('+inf')
        )
        dist_cdr, cdr2antigen_min_idx = torch.topk(
            dist_cdr, 
            k = nb_hotspots,
            largest=False, sorted=True
        )    # (nb_hotspots, )
        if self.dist_cutoff is not None and dist_cdr.min() > self.dist_cutoff:
            raise CUTOFFVIOLATION(f'No interaction pairs found within cutoff of {self.dist_cutoff}')
        # antibody-antigen interaction info 
        # the antigen residues that has cloest distance to the antibody residues
        data['antigen_hotspots'] = torch.ones_like(data['generate_flag'], dtype=torch.int64) 
        data['antigen_hotspots'][cdr2antigen_min_idx] = 2 
        
        # the cloest distance
        data['to_hotspot_dist'] = torch.zeros_like(data['generate_flag'], dtype=torch.float32)
        data['to_hotspot_dist'][cdr2antigen_min_idx] = dist_cdr
        
        hotspot_label = torch.ones_like(data['generate_flag'], dtype=torch.int64)
        hotspot_distance = torch.ones_like(data['generate_flag'], dtype=torch.long)
        if self.max_nb_hotspots > 0:
            # 2 for hotspot, 1 for non-hotspot. 0 saved for padding
            hotspot_label[cdr2antigen_min_idx] = 2 
            hotspot_distance[cdr2antigen_min_idx] = torch.floor(dist_cdr).long().clamp(1, self.max_distance-1)
        data['hotspot_label'] = hotspot_label
        data['hotspot_distance'] = hotspot_distance
        ### patch ###
        patch_mask = torch.logical_or(
            data['generate_flag'],
            data['anchor_flag'],
        )
        patch_mask[initial_patch_idx] = True
        patch_mask[antigen_patch_idx] = True

        patch_idx = torch.arange(0, patch_mask.shape[0])[patch_mask]

        data_patch = _mask_select_data(data, patch_mask)
        data_patch = self._center(
            data_patch,
            origin = anchor_points.mean(dim=0)
        )
        data_patch['patch_idx'] = patch_idx
        
        return data_patch
