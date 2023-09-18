import torch
import torch.nn as nn
from typing import Optional, Callable, List, Tuple, Sequence

from src.modules.common.layers import DistanceToBins, LayerNorm
from src.modules.common.nn import Linear, PerResiduePredictor
    
class PerResidueRMSDCaPredictor(PerResiduePredictor):
    def __init__(self, no_bins, c_in, c_hidden):
        super().__init__(no_bins, c_in, c_hidden)

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

class pRMSDCa(nn.Module):
    def __init__(self, num_bins=20, dist_min=0.5, dist_max=19.5):
        """
        [dist_min, ..., dist_max]
        """
        super(pRMSDCa, self).__init__()

        self.num_bins = num_bins
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.tobin = DistanceToBins(dist_min=self.dist_min, dist_max=self.dist_max, num_bins=self.num_bins, use_onehot=True)
        
    def compute_prmsd(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): [..., num_bins]

        Returns:
            torch.Tensor: [...,]
        """
        num_bins = logits.shape[-1]
        assert num_bins == self.num_bins
        bounds = torch.linspace(self.dist_min, self.dist_max, self.num_bins, device=logits.device)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_rmsd_ca = torch.sum(
            probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
            dim=-1,
        )
        return pred_rmsd_ca

    def forward(self,prmsd_logits: torch.Tensor, rmsd: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self.calc_prmsd_loss(prmsd_logits, rmsd, mask)
        return loss
    
    def calc_prmsd_loss(self, prmsd_logits: torch.Tensor, rmsd: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prmsd_logits (torch.Tensor): [B, ..., num_logits]
            rmsd (torch.Tensor): [B, ...]
            mask (torch.Tensor): [B, ...]
        
        Returns:
            torch.Tensor: 
        """
        eps: float = 1e-10

        
        rmsd_onehot = self.tobin(rmsd.unsqueeze(-1),dim=-1)
        errors = softmax_cross_entropy(prmsd_logits, rmsd_onehot) # [B, ...]
        loss = (errors * mask).sum() / (mask.sum() + eps)
        return loss

    @staticmethod
    def calc_per_rmsd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): [B, L, 3]
            target (torch.Tensor): [B, L, 3]

        Returns:
            torch.Tensor: [B, L]
        """
        diff = pred - target
        diff = diff ** 2
        diff = torch.sum(diff, dim=-1)
        diff = torch.sqrt(diff)
        return diff

    @staticmethod
    def calc_rmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): [B, L, 3]
            target (torch.Tensor): [B, L, 3]
            mask (torch.Tensor): [B, L]

        Returns:
            torch.Tensor: [B]
        """
        # Apply mask to predicted and target tensors
        pred_masked = pred * mask.unsqueeze(-1)
        target_masked = target * mask.unsqueeze(-1)

        # Calculate squared distance between masked predicted and target tensors
        squared_diff = torch.sum((pred_masked - target_masked)**2, dim=-1)

        # Count number of unmasked positions
        num_unmasked = torch.sum(mask, dim=-1)

        # Calculate RMSD for each example
        rmsd = torch.sqrt(torch.sum(squared_diff, dim=-1) / num_unmasked)

        return rmsd

