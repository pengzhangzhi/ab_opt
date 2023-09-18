import math
import torch
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
from torch_scatter import scatter_sum

DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_id': ' ', 
    'icode': ' ',
}

DEFAULT_NO_PADDING = {
    'origin',
}

class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)
        
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k)) if k not in self.no_padding else v
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def apply_patch_to_tensor(x_full, x_patch, patch_idx):
    """
    Args:
        x_full:  (N, ...)
        x_patch: (M, ...)
        patch_idx:  (M, )
    Returns:
        (N, ...)
    """
    x_full = x_full.clone()
    x_full[patch_idx] = x_patch
    return x_full

class CollateNodes(object):
    def __init__(self,*args, **kwargs): 
        self.padding_collate = PaddingCollate(*args, **kwargs)
    def __call__(self, data_list):
        default_batch = self.padding_collate(data_list)
        coord_list = []
        aa_list = []
        batch_id_list = []
        fragment_id_list = []
        generate_flag_range_list = []
        for i,data in enumerate(data_list):
            seq_len = data['aa'].shape[0]
            coord = data['pos_heavyatom']
            fragment_type = data['fragment_type']
            batch_id = torch.full((seq_len,), i, dtype=torch.long)
            flag = torch.arange(seq_len)[data['generate_flag']]
            assert (flag[1:] - flag[:-1] == 1).all(), 'generate_flag must be continuous' 
            generate_flag_range_list.append(torch.tensor([flag[0], flag[-1]]))
            coord_list.append(coord)
            aa_list.append(data['aa'])
            batch_id_list.append(batch_id)  
            fragment_id_list.append(fragment_type)
        
        coord_tns = torch.cat(coord_list, dim=0) # [nb_all_residues, nb_atoms, 3]
        aa_tns = torch.cat(aa_list, dim=0) # [nb_all_residues]
        batch_id_tns = torch.cat(batch_id_list, dim=0)
        fragment_id_tns = torch.cat(fragment_id_list, dim=0)
        lengths = scatter_sum(torch.ones_like(batch_id_tns), batch_id_tns)  # [bs]
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # [bs, 2]
        generate_flag_range_tns = torch.stack(generate_flag_range_list, dim=0) + offsets.unsqueeze(-1)
        
        default_batch['graph'] = {
            'coord': coord_tns,
            'batch_id': batch_id_tns,
            'fragment_id': fragment_id_tns,
            "genearte_flag_range": generate_flag_range_tns,
            "aa": aa_tns,
        }
        return default_batch