import torch

from ._base import _mask_select_data, register_transform
from ..protein import constants

class DISTVIOLATION(Exception):
    pass
@register_transform('patch_around_anchor')
class PatchAroundAnchor(object):

    def __init__(
        self,
        initial_patch_size=128,
        antigen_size=128,
        remove_anchor=False,
        dist_cutoff=None,
        crop_contiguous_antigen=False,
        contiguous_threshold=1e6,
        contiguous_ratio=0.0,
        ):
        super().__init__()
        self.initial_patch_size = int(initial_patch_size)
        self.antigen_size = int(antigen_size)
        self.remove_anchor = remove_anchor if isinstance(remove_anchor,bool) else remove_anchor.lower() not in ['false', '0']
        self.dist_cutoff = dist_cutoff
        self.crop_contiguous_antigen = crop_contiguous_antigen if isinstance(crop_contiguous_antigen,bool) else crop_contiguous_antigen.lower() not in ['false', '0']
        self.contiguous_threshold = int(contiguous_threshold)
        self.contiguous_ratio = float(contiguous_ratio)
        
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
        cdr_points = data['pos_heavyatom'][data['generate_flag']][:, constants.BBHeavyAtom.CA]  # (cdr_len, 3)
        dist_anchor = torch.cdist(pos_alpha, cdr_points).min(dim=1)[0]    # (L, )
        initial_patch_idx = torch.topk(
            dist_anchor,
            k = min(self.initial_patch_size, dist_anchor.size(0)),
            largest=False,
        )[1]   # (initial_patch_size, )

        dist_anchor_antigen = dist_anchor.masked_fill(
            mask = antibody_mask, # Fill antibody with +inf
            value = float('+inf')
        )   # (L, )
        min_dist = dist_anchor_antigen.min().item()
        if (self.dist_cutoff is not None) and \
            (min_dist > 0) and \
            (min_dist > self.dist_cutoff):
            raise DISTVIOLATION('Distance between anchor and antigen is too large')
        antigen_patch_idx = torch.topk(
            dist_anchor_antigen, 
            k = min(self.antigen_size, antigen_mask.sum().item()), 
            largest=False, sorted=True
        )[1]    # (ag_size, )
        if self.crop_contiguous_antigen and antigen_patch_idx.shape[0] > 0:
            contiguous_ratio, antigen_patch_idx = get_contiguous_idx(antigen_patch_idx, self.contiguous_threshold)
            if contiguous_ratio < self.contiguous_ratio:
                raise DISTVIOLATION(f'Antigen is not contiguous, keep ratio: {contiguous_ratio}')
        patch_mask = data['generate_flag'].clone()
        if not self.remove_anchor:
            patch_mask = torch.logical_or(
                patch_mask,
                anchor_flag,
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


def get_contiguous_idx(idx_tns, theshold=3):
    """
    Check if the indices are contiguous
    Args:
        idx_tns (torch.Tensor): indices tensor, shape (N, )
    Returns:
        contiguous_ratio (float): the ratio of the contiguous indices
        idx_tns (torch.Tensor): the indices tensor after removing the non-contiguous indices
    """
    idx_tns = idx_tns.sort()[0]
    diff = idx_tns[1:] - idx_tns[:-1]
    contiguous_tensor = torch.cat([torch.tensor([True]), diff <= theshold])
    # find the index of the first False value
    if contiguous_tensor.sum().item() == len(contiguous_tensor):
        return 1, idx_tns
    
    first_false_idx = torch.where(contiguous_tensor == False)[0][0]
    last_false_idx = torch.where(contiguous_tensor == False)[0][-1]
    if len(idx_tns) - 1 - last_false_idx > first_false_idx - 0:
        contiguous_tensor[:last_false_idx] = False
    else:
        contiguous_tensor[first_false_idx:] = False
    contiguous_ratio = round(contiguous_tensor.sum().item()/ len(contiguous_tensor),2)
    return contiguous_ratio, idx_tns[contiguous_tensor]