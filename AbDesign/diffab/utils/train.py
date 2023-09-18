import pandas as pd
from collections import defaultdict
import json
import os
import numpy as np
import torch
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from easydict import EasyDict
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.protein.writers import save_pdb
from diffab.utils.transforms import Compose

from diffab.utils.data import CollateNodes, PaddingCollate
from diffab.utils.inference import RemoveNative
from diffab.utils.transforms.label import Label
from diffab.utils.transforms.merge import MergeChains
from diffab.utils.transforms.patch import PatchAroundAnchor
from diffab.utils.val import create_data_variants,run_on_variant
from diffab.utils.protein.constants import char2hydropathy,char2charge
from .misc import BlackHole, save_each_sample


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    elif cfg.type is None:
        return BlackHole()
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def get_warmup_sched(cfg, optimizer):
    if cfg is None: return BlackHole()
    lambdas = [lambda it : (it / cfg.max_iters) if it <= cfg.max_iters else 1 for _ in optimizer.param_groups]
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return warmup_sched


def log_losses(out, it, tag, logger=BlackHole(), writer=BlackHole(), others={}):
    logstr = '[%s] Iter %05d' % (tag, it)
    logstr += ' | loss %.4f' % out['overall'].item()
    for k, v in out.items():
        if k == 'overall': continue
        if isinstance(v, torch.Tensor): v = v.item()
        logstr += ' | loss(%s) %.4f' % (k, v)
    for k, v in others.items():
       logstr += ' | %s %2.4f' % (k, v)
    logger.info(logstr)

    for k, v in out.items():
        if k == 'overall':
            writer.add_scalar('%s/loss' % tag, v, it)
        else:
            writer.add_scalar('%s/loss_%s' % (tag, k), v, it)
    for k, v in others.items():
        writer.add_scalar('%s/%s' % (tag, k), v, it)
    writer.flush()


class ValidationLossTape(object):

    def __init__(self):
        super().__init__()
        self.accumulate = {}
        self.others = {}
        self.total = 0

    def update(self, out, n, others={}):
        self.total += n
        for k, v in out.items():
            if k not in self.accumulate:
                self.accumulate[k] = v.clone().detach()
            else:
                self.accumulate[k] += v.clone().detach()

        for k, v in others.items():
            if k not in self.others:
                self.others[k] = v.clone().detach()
            else:
                self.others[k] += v.clone().detach()
        

    def log(self, it, logger=BlackHole(), writer=BlackHole(), tag='val'):
        avg = EasyDict({k:v / self.total for k, v in self.accumulate.items()})
        avg_others = EasyDict({k:v / self.total for k, v in self.others.items()})
        log_losses(avg, it, tag, logger, writer, others=avg_others)
        return avg['overall']


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


def reweight_loss_by_sequence_length(length, max_length, mode='sqrt'):
    if mode == 'sqrt':
        w = np.sqrt(length / max_length)
    elif mode == 'linear':
        w = length / max_length
    elif mode is None:
        w = 1.0
    else:
        raise ValueError('Unknown reweighting mode: %s' % mode)
    return w


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    if weights is None:
        # initialize weights to 1
        weights = {k:1.0 for k in losses.keys()}
    loss = 0
    for k in losses.keys():
        if not (isinstance(losses[k],torch.Tensor) and losses[k].requires_grad is True):
            # skip non-tensor or non-grad losses
            continue
        if weights.get(k) is None:
            weights[k] = 1
        loss = loss + weights[k] * losses[k]
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def eval_sample(config, get_structure, model, logger, save_dir, hydropathy_spec=None,charge_spec=None,  **kwargs):
    model.eval()
    sample_id = get_structure()['id']
    logger.info('Data ID: %s' % sample_id)

    sample_dir = os.path.join(save_dir, f"{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)
    
    data_variants = create_data_variants(
        config = config,
        structure_factory = get_structure,
    )
    metadata = {
        'identifier': sample_id,
        'config': config,
        'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
    }
    with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    inference_tfm = [ 
                        PatchAroundAnchor(
                            config.antigen_size, 
                            config.initial_patch_size,
                            max_nb_hotspots=config.max_nb_hotspots,
                            is_train=False
                            ), 
                        Label(
                            hydropathy=config.hydropathy,
                            charge=config.charge,
                            drop_prob=config.drop_prob,
                        )
                    ]
    if 'abopt' not in config.mode:  # Don't remove native CDR in optimization mode
        inference_tfm.append(RemoveNative(
            remove_structure = config.sampling.sample_structure,
            remove_sequence = config.sampling.sample_sequence,
        ))
    inference_tfm = Compose(inference_tfm)
    
    data_native = MergeChains()(get_structure())
    config.sampling.num_samples = 1
    for variant in data_variants:
        os.makedirs(os.path.join(sample_dir, variant['tag']), exist_ok=True)
        logger.info(f"Start sampling for: {variant['tag']}")
        save_pdb(data_native, os.path.join(sample_dir, variant['tag'], 'REF1.pdb'))       # w/  OpenMM minimization
        
        data_cropped = inference_tfm(
                copy.deepcopy(variant['data'])
        )
        print('hydropathy_spec', hydropathy_spec)
        if hydropathy_spec is not None:
            hydropathy_labels = data_cropped['hydropathy'][data_cropped['generate_flag']]
            print("before:", hydropathy_labels)
            for idx, label in hydropathy_spec.items():
                hydropathy_labels[int(idx)] = char2hydropathy[str(label)]
            data_cropped['hydropathy'][data_cropped['generate_flag']] = hydropathy_labels
            print("after:", data_cropped['hydropathy'][data_cropped['generate_flag']])
        if charge_spec is not None:
            charge_labels = data_cropped['charge'][data_cropped['generate_flag']]
            for idx, label in charge_spec.items():
                charge_labels[int(idx)] = char2charge[str(label)]
            data_cropped['charge'][data_cropped['generate_flag']] = charge_labels
        batch_size = min(100, config.sampling.num_samples)
        assert config.sampling.num_samples == 1, 'sample once and save all the trajectories'
        data_list_repeat = [ data_cropped ] * config.sampling.num_samples
        collate_fn = CollateNodes(eight=False)
        loader = DataLoader(data_list_repeat, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
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
            traj_pdb_str = ''
            for traj_idx in (traj_batch):
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
                result,pdb_path = save_each_sample(sample_dir, variant, data_cropped, traj_idx, batch, aa_new, pos_atom_new, mask_atom_new, i=0)
                result = {variant['tag']+"-"+k:torch.tensor(v).mean().item() if isinstance(v, (torch.Tensor, float)) else v for k, v in result.items() }
                with open(pdb_path, 'r') as f:
                    traj_pdb_str += f'MODEL     {traj_idx+1}'+f.read()+'ENDMDL\n'
                    
            traj_path = os.path.join(sample_dir, f"traj.pdb")
            with open(traj_path, 'w') as f:
                f.write(traj_pdb_str)
                print(f"Saved trajectory to {traj_path}")
            logger.info(f"traj: {traj_idx} "+", ".join(f"{k}: {v:.2f}" if isinstance(v,float) else f"{k}: {v}" for k, v in result.items()))
        

@torch.no_grad()
def eval_on_dataset(config, dataset, model, logger, save_dir):
    """ evaluate on val or test set. """
    datset_size = len(dataset)
    model.eval()
    dataset_results = defaultdict(list)
    id_lst = []
    for i in range(datset_size):
        # evaluate each sample
        get_structure = lambda: dataset[i]
        sample_id = get_structure()['id']
        logger.info('Data ID: %s' % sample_id)
    
        sample_dir = os.path.join(save_dir, f"{str(i)}-{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)
        
        data_variants = create_data_variants(
            config = config,
            structure_factory = get_structure,
        )
        metadata = {
            'identifier': sample_id,
            'index': i,
            'config': config,
            'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
        }
        with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        inference_tfm = [ 
                         PatchAroundAnchor(
                             config.antigen_size, 
                             config.initial_patch_size,
                             max_nb_hotspots=config.max_nb_hotspots,
                             is_train=False
                             ), 
                        Label(
                            hydropathy=config.hydropathy, charge=config.charge,drop_prob=config.drop_prob,
                        )
                        ]
        if 'abopt' not in config.mode:  # Don't remove native CDR in optimization mode
            inference_tfm.append(RemoveNative(
                remove_structure = config.sampling.sample_structure,
                remove_sequence = config.sampling.sample_sequence,
            ))
        inference_tfm = Compose(inference_tfm)
        
        data_native = MergeChains()(get_structure())
        for variant in data_variants:
            result = run_on_variant(config, sample_dir, logger, data_native, model, inference_tfm, variant)
            
            result = {variant['tag']+"-"+k:torch.tensor(v).mean().item() if isinstance(v[0],(float,torch.Tensor)) else v[-1] for k, v in result.items()}
            
            logger.info(f"idx: {i}, "+", ".join(f"{k}: {v:.2f}" if isinstance(v,float) else f"{k}: {v}" for k, v in result.items()))
            for k,v in result.items():
                dataset_results[k].append(v)
                
        if len(data_variants) != 0: id_lst.append(sample_id)
        
    df = pd.DataFrame(dataset_results, index=id_lst)
    # df.loc['mean'] = df.mean()
    df.to_csv(os.path.join(save_dir, 'results.csv'), float_format='%.2f')
    print(df)
    return dict(df.mean())