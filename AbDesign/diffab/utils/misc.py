from diffab.utils.data import apply_patch_to_tensor
from diffab.utils.protein.constants import BBHeavyAtom,ID2RES1,hydropathy2char,charge2char
import os
import time
import random
import logging
from typing import OrderedDict
import torch
import torch.linalg
import numpy as np
import yaml
from easydict import EasyDict
from glob import glob
import torch.nn.functional as F

from diffab.utils.protein.writers import save_pdb

class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class Counter(object):
    def __init__(self, start=0):
        super().__init__()
        self.now = start

    def step(self, delta=1):
        prev = self.now
        self.now += delta
        return prev


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def get_checkpoint_path(folder, it=None):
    if it is not None:
        return os.path.join(folder, '%d.pt' % it), it
    all_iters = list(map(lambda x: int(os.path.basename(x[:-3])), glob(os.path.join(folder, '*.pt'))))
    all_iters.sort()
    return os.path.join(folder, '%d.pt' % all_iters[-1]), all_iters[-1]


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


def extract_weights(weights: OrderedDict, prefix):
    extracted = OrderedDict()
    for k, v in weights.items():
        if k.startswith(prefix):
            extracted.update({
                k[len(prefix):]: v
            })
    return extracted


def current_milli_time():
    return round(time.time() * 1000)




def batchfy(nodes, lengths):
    """
    convert a sequence of samples into a structured batch.
    Args:
        nodes: (N, ...)
        lengths: (B, ), the length of each sample in the batch.
    Returns:
        batch: (B, n, ... ), n is the maximum length of the samples.
    """
    assert lengths.sum() == nodes.shape[0]
    n = lengths.max()
    b = len(lengths)
    # (B, n, ...)
    out = torch.zeros(
        (b, n, *nodes.shape[1:]), 
        dtype=nodes.dtype, device=nodes.device
        )
    offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)
    for i, l in enumerate(lengths):
        offset = offsets[i]
        out[i, :l] = nodes[offset:offset+l]
    return out


def pair2edge(edge_idx, lengths, pair):
    """
    convert pair to edges.

    Args:
        edge_idx (torch.Tensor): [2, N]
        pair (torch.Tensor): [B, L, L, D]
        lengths (torch.Tensor): [B, ]
    Returns:
        edge (torch.Tensor): [N, D]
    """
    # [B+1, ] the offsets of each sample in the batch, pad zero at the beginning.
    B = len(lengths)
    offsets = F.pad(torch.cumsum(lengths, dim=0), pad=(1, 0), value=0)
    edge_feats = []
    for i in range(B):
        prev_offset = offsets[i]
        next_offset = offsets[i+1]
        idx = (edge_idx >= prev_offset) & (edge_idx < next_offset)
        local_edge_idx = edge_idx[idx].reshape(2,-1) - prev_offset
        edge_feat = pair[i][local_edge_idx[0],local_edge_idx[1]]
        edge_feats.append(edge_feat)
    edge_feats = torch.cat(edge_feats,dim=0)
    
    return edge_feats


def clash_loss(positions, seq_mask, chain_id, lit=3.0078, ep = 1e-6):
    """
    The clash loss penalizes too short distances only for residues from different chain.


    Args:
        positions (torch.Tensor): [B, N, 3]
        seq_mask (torch.Tensor): [B, N]
        chain_id (torch.Tensor): [B, N] the chain id of each residue
        lit (float): average shortest cdr to antigen distance. 
                    Defaults to 3.0078, which is calculated from the training set.
                    
        tolerance (float, optional): set to the standard deviation of the cdr2antigen distance.
                                    Defaults to 3.3694.
    Returns:
        loss (torch.Tensor): 
    """
    # [B, N, N]
    pair_different_chain_mask = chain_id.unsqueeze(1) != chain_id.unsqueeze(2) 
    pair_mask = seq_mask.unsqueeze(1) * seq_mask.unsqueeze(2)
    mask = pair_mask.float() * pair_different_chain_mask.float()
    pair_dist = torch.cdist(positions, positions, p=2)
    pair_loss = F.relu(lit-pair_dist) * mask
    loss = pair_loss.sum() / ((pair_loss>0).sum() + ep) # average clash pairs
    # print(((pair_loss>0).sum(-1).sum(-1)/2).mean())
    # print(loss*10)
    return loss


def hotspot_distance_fn(x, hotspot_idx, cdr_idx,):
    """
    return the distance between cdr residues and the hotspots.
    

    Args:
        x (torch.Tensor): [B, N, 3]
        hotspot_idx (torch.Tensor): [B, X], X is 
        the number of hotspots
        cdr_idx (torch.Tensor): [B, Y], Y is the number of cdr residues
    Returns:
        pairwsie distance, [B, X, Y],
    """
    dist_mat = torch.cdist(x, x)
    cdr2hotspot_dist = dist_mat[:,cdr_idx, hotspot_idx]
    # cdr2hotspot_dist = dist_mat.gather(2, cdr_idx.unsqueeze(-1)).gather(1, hotspot_idx.unsqueeze(-1))
    return cdr2hotspot_dist
    

def save_each_sample(log_dir, variant, data_cropped, count, batch, aa_new, pos_atom_new, mask_atom_new, i):
    data_tmpl = variant['data']
    aa = apply_patch_to_tensor(data_tmpl['aa'], aa_new[i], data_cropped['patch_idx'])
    mask_ha = apply_patch_to_tensor(data_tmpl['mask_heavyatom'], mask_atom_new[i], data_cropped['patch_idx'])
    pos_ha  = (
                    apply_patch_to_tensor(
                        data_tmpl['pos_heavyatom'], 
                        pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(), 
                        data_cropped['patch_idx']
                    )
                )

    save_path = os.path.join(log_dir, variant['tag'], '%04d.pdb' % (count, ))
    save_pdb({
        'chain_nb': data_tmpl['chain_nb'],
        'chain_id': data_tmpl['chain_id'],
        'resseq': data_tmpl['resseq'],
        'icode': data_tmpl['icode'],
        # Generated
        'aa': aa,
        'mask_heavyatom': mask_ha,
        'pos_heavyatom': pos_ha,
    }, path=save_path)
    generate_flags = data_tmpl["generate_flag"]
    native_atom_positions = data_tmpl["pos_heavyatom"][
                            ..., BBHeavyAtom.CA, :
                        ][generate_flags]
    pred_atom_positions = pos_ha[..., BBHeavyAtom.CA, :][generate_flags]
    rmsd = torch.sqrt(torch.mean(torch.sum(((native_atom_positions - pred_atom_positions) ** 2), dim=1)))
                
    native_aa = data_tmpl['aa'][generate_flags]
    pred_aa = aa[generate_flags]
    aa_recovery = torch.sum(native_aa == pred_aa) / native_aa.size(0)
    aa_seq = "".join([ID2RES1[int(aa_idx)] for aa_idx in pred_aa])
    native_aa_seq = "".join([ID2RES1[int(aa_idx)] for aa_idx in native_aa])
    hydropathy = "".join([hydropathy2char[int(label)] for label in data_cropped['hydropathy'][data_cropped['generate_flag']]])
    charge =  "".join([charge2char[int(label)] for label in data_cropped['charge'][data_cropped['generate_flag']]])
    # print(f"RMSD: {rmsd:.2f}, AA recovery: {aa_recovery:.2f}")
    save_pdb({
            'chain_nb': data_cropped['chain_nb'],
            'chain_id': data_cropped['chain_id'],
            'resseq': data_cropped['resseq'],
            'icode': data_cropped['icode'],
            # Generated
            'aa': aa_new[i],
            'mask_heavyatom': mask_atom_new[i],
            'pos_heavyatom': pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(),
        }, 
        path=os.path.join(log_dir, variant['tag'], '%04d_patch.pdb' % (count, )))

    return {
        "rmsd": rmsd.item(),
        "aa_recovery": aa_recovery.item(),
        "aa_seq":aa_seq,
        "native_aa_seq": native_aa_seq,
        "hydropathy": hydropathy,
        "charge": charge,
    }, save_path