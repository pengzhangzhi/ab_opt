import os
import shutil
import argparse
import torch
import glob
# import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train/codesign_single.yml')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None, help='ckpt path. Resume from a checkpoint.')
    parser.add_argument('--finetune', type=str, default=None, help='ckpt path. Finetune from a pretrained model with extra finetune_iters.')
    parser.add_argument('--finetune_iters', type=int, default=200_000)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    if os.environ.get('wandb') is not None:
        import wandb
        wandb.init(project="diffusion-antibody", config=dict(config), name=f"{args.tag}-{config_name}")
    
    
    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = BlackHole() # torch.utils.tensorboard.SummaryWriter(log_dir)
        # tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    train_iterator = inf_iterator(DataLoader(
        train_dataset, 
        batch_size=config.train.batch_size, 
        collate_fn=CollateNodes(),
        shuffle=True,
        num_workers=args.num_workers
    ))
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=CollateNodes(), shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
        

    # Train
    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)

        # Forward
        # if args.debug: torch.set_anomaly_enabled(True)
        loss_dict = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        time_forward_end = current_milli_time()

        # Backward
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        log_losses(loss_dict, it, 'train', logger, writer, others={
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        if os.environ.get('wandb') is not None:
            track_dict = {
                **{k: float(v) for k, v in loss_dict.items()},
                **{
                    "grad": orig_grad_norm.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                },
            }
            wandb.log(track_dict)
            
        if not torch.isfinite(loss):
            logger.error('NaN or Inf detected.')
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': it,
                'batch': recursive_to(batch, 'cpu'),
            }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
            raise KeyboardInterrupt()

    # Validate
    def validate(it):
        save_dir = os.path.join(log_dir, 'val', str(it))
        os.makedirs(save_dir, exist_ok=True)
        val_result = eval_on_dataset(config, val_dataset, model, logger, save_dir)
        # find a file ends with pdb in the subdirectory of save_dir recursively
        if os.environ.get('wandb') is not None:
            wandb.log(val_result)
            pdb_file = glob(os.path.join(save_dir, '*', '*', '*patch.pdb'))[0]
            wandb.log({'val_sample': wandb.Molecule(pdb_file)})
            
        return val_result
    max_iters = it_first+args.finetune_iters if args.finetune is not None else config.train.max_iters
    min_rmsd = 1e6
    try:
        # validate(0)
        for it in range(it_first, max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                val_result = validate(it)
                if not args.debug:
                    if val_result['H_CDR3-rmsd'] < min_rmsd:
                        min_rmsd = val_result['H_CDR3-rmsd']
                        aar = val_result['H_CDR3-aa_recovery']
                        fname = f'best-{it}-RMSD-{float(min_rmsd):.3f}-AAR-{float(aar):.3f}.pt'
                    else: fname = '%d.pt' % it
                    ckpt_path = os.path.join(ckpt_dir, fname)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'val_result': val_result,
                    }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
