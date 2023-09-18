# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from src.modules.common.nn import PerResiduePredictor
from src.modules.common.prmsd import softmax_cross_entropy

from src.modules.common.tensor_utils import permute_final_dims

class PerResidueLDDTCaPredictor(PerResiduePredictor):
    def __init__(self, no_bins, c_in, c_hidden):
        super().__init__(no_bins, c_in, c_hidden)
        
def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score



def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    """
    Compute the LDDT loss.

    Args:
        logits (torch.Tensor): A tensor of shape [B, L, no_bins],
        all_atom_pred_pos (torch.Tensor): A tensor of shape [B, L, 3], 
        all_atom_positions (torch.Tensor): A tensor of shape [B, L, 3], 
        all_atom_mask (torch.Tensor): A binary mask of shape [B, L, 1], where each element is either 0 or 1, 
            indicating whether the corresponding atom in the sequence should be included in the calculation 
            of the LDDT loss.
        cutoff (float, optional): The distance cutoff (in Angstroms) beyond which pairwise distances are 
            considered to be meaningless and ignored in the calculation of the LDDT loss. Defaults to 15.0.
        no_bins (int, optional): 

    Returns:
        torch.Tensor: _description_
    """
    score = lddt(
        all_atom_pred_pos, 
        all_atom_positions, 
        all_atom_mask, 
        cutoff=cutoff, 
        eps=eps
    )

    score = score.detach()

    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
        eps + torch.sum(all_atom_mask, dim=-1)
    )


    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss