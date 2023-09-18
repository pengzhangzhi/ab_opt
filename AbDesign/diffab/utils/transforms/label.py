import torch

from ._base import _mask_select_data, register_transform
from ..protein.constants import Hydropathy, Charge, aaidx2charge,aaidx2hydropathy


@register_transform('aa_label')
class Label(object):

    def __init__(self, hydropathy=False, charge=False,drop_prob=0.0):
        super().__init__()
        self.hydropathy = hydropathy
        self.charge = charge
        self.drop_prob = drop_prob
    def __call__(self, data):
        aa_flags = data['generate_flag']
        aa = data['aa']
        
        hydropathy_flags = torch.zeros_like(aa).fill_(Hydropathy.unknown)
        if self.hydropathy:
            hydropathy_label = torch.tensor([aaidx2hydropathy[int(aaidx)] for aaidx in aa[aa_flags]],dtype=torch.int64)
            # randomly drop some labels with probability drop_prob
            if self.drop_prob > 0.0:
                drop_mask = torch.rand_like(hydropathy_label.float()) < self.drop_prob 
                hydropathy_label[drop_mask] = Hydropathy.unknown
            hydropathy_flags[aa_flags] = hydropathy_label
        data['hydropathy'] = hydropathy_flags
        
        charge_flags = torch.zeros_like(aa).fill_(Charge.unknown)
        if self.charge:
            charge_label = torch.tensor([aaidx2charge[int(aaidx)] for aaidx in aa[aa_flags]], dtype=torch.int64)
            # randomly drop some labels with probability drop_prob
            if self.drop_prob > 0.0:
                drop_mask = torch.rand_like(charge_label.float()) < self.drop_prob
                charge_label[drop_mask] = Charge.unknown
            charge_flags[aa_flags] = charge_label
        data['charge'] = charge_flags
        return data