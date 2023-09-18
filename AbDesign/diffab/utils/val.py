from collections import defaultdict
import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.inference import RemoveNative
from diffab.utils.protein.constants import BBHeavyAtom
from diffab.utils.protein.writers import save_pdb
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.transforms import *
from diffab.utils.inference import *


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        if device == 'cpu':
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def create_data_variants(config, structure_factory):
    structure = structure_factory()
    structure_id = structure['id']

    data_variants = []
    if config.mode == 'single_cdr':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            data_variants.append({
                'data': data_var,
                'name': f'{structure_id}-{cdr_name}',
                'tag': f'{cdr_name}',
                'cdr': cdr_name,
                'residue_first': residue_first,
                'residue_last': residue_last,
            })
    elif config.mode == 'multiple_cdrs':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-MultipleCDRs',
            'tag': 'MultipleCDRs',
            'cdrs': cdrs,
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'full':
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-Full',
            'tag': 'Full',
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'abopt':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    'data': data_var,
                    'name': f'{structure_id}-{cdr_name}-O{opt_step}',
                    'tag': f'{cdr_name}-O{opt_step}',
                    'cdr': cdr_name,
                    'opt_step': opt_step,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                })
    else:
        raise ValueError(f'Unknown mode: {config.mode}.')
    return data_variants


def run_on_variant(config, log_dir, logger, data_native, model, inference_tfm, variant,**kwargs):
    """
    return:
        result_dict: dict, keys are rmsd, aa_recovery, aa_seq, value is lists. 
    """
    
    os.makedirs(os.path.join(log_dir, variant['tag']), exist_ok=True)
    logger.info(f"Start sampling for: {variant['tag']}")

    save_pdb(data_native, os.path.join(log_dir, variant['tag'], 'REF1.pdb'))       # w/  OpenMM minimization
    
    data_cropped = inference_tfm(
            copy.deepcopy(variant['data'])
    )
    batch_size = min(100, config.sampling.num_samples)
    data_list_repeat = [ data_cropped ] * config.sampling.num_samples
    collate_fn = CollateNodes(eight=False)
    loader = DataLoader(data_list_repeat, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
    count = 0
    result_dict = defaultdict(list)
    for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
        # batch size
        count, result = run_on_batch(config, log_dir, model, variant, data_cropped, count, batch,**kwargs)
        count += batch_size
        for k, v in result.items():
            result_dict[k].extend(v)
        
    logger.info('Finished.\n')
    return result_dict


def run_on_batch(config, log_dir, model, variant, data_cropped, count, batch, traj_idx = 0):
    torch.set_grad_enabled(False)
    model.eval()
    device = next(model.parameters()).device
    batch = recursive_to(batch, device)
    if 'abopt' in config.mode:
                # Antibody optimization starting from native
        traj_batch = model.optimize(batch, opt_step=variant['opt_step'], optimize_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })
    else:
                # De novo design
        traj_batch = model.sample(batch,
                sample_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                },
                
                )
    
    result_dict = save_trajectory(log_dir, variant, data_cropped, count, batch, traj_idx, device, traj_batch)
    return count, result_dict

def save_trajectory(log_dir, variant, data_cropped, count, batch, traj_idx, device, traj_batch):
    """
    return:
        result_dict: dict, keys are rmsd, aa_recovery, aa_seq, value is lists. 
    """
    r_new = so3vec_to_rotation(traj_batch[traj_idx][0]).to(device)
    t_new = (traj_batch[traj_idx][1]).to(device)
    aa_new = (traj_batch[traj_idx][2]).to(device)   # 2: Amino acid.
    pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx = batch['pos_heavyatom'],
                R_new = r_new,
                t_new = t_new,
                aa = aa_new,
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms = batch['mask_heavyatom'],
                mask_recons = batch['generate_flag'],
            )
    aa_new = aa_new.cpu()
    pos_atom_new = pos_atom_new.cpu()
    mask_atom_new = mask_atom_new.cpu()
    result_dict = defaultdict(list)
    for i in range(aa_new.size(0)):
        result,pdb_path = save_each_sample(log_dir, variant, data_cropped, count, batch, aa_new, pos_atom_new, mask_atom_new, i)
        count += 1
        # append items in result to result_dict
        for k, v in result.items():
            result_dict[k].append(v)
    return result_dict


