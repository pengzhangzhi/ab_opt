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
from diffab.tools.eval.run import run_energy
from diffab.tools.relax.run import run_relax
from diffab.utils.inference import RemoveNative
from diffab.utils.protein.constants import BBHeavyAtom
from diffab.utils.protein.writers import save_pdb
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import eval_on_dataset,eval_sample
from diffab.utils.transforms import *
from diffab.utils.inference import *
from diffab.utils.val import create_data_variants, run_on_variant



def main():
    args = get_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Testset
    dataset = get_dataset(config.dataset.test)

    # Logging
    tag_postfix = '_%s' % args.tag if args.tag else ''
    # log_dir = get_new_log_dir(os.path.join(args.out_root, config_name + tag_postfix), prefix='%04d_%s' % (args.index, structure_['id']))
    logger = get_logger('sample')
    logger.info('Loading sampling configs: %s' % (args.config))
    ckpt_path = args.ckpt if args.ckpt is not None else config.model.checkpoint
    # Load checkpoint and model
    logger.info('Loading model config and checkpoints: %s' % (ckpt_path))
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    logger.info(str(lsd))

    save_dir = os.path.join(args.out_root, config_name + tag_postfix)
    if args.index is not None:
        get_structure = lambda: dataset[args.index]
        eval_sample(
            config, get_structure, model, logger, save_dir, 
            traj_idx=args.traj_idx, hydropathy_spec=args.hydropathy_spec, charge_spec=args.charge_spec
            )
    else:
        eval_on_dataset(config, dataset, model, logger, save_dir)
        run_relax(save_dir,pipeline_name='openmm_pyrosetta')
        run_energy(save_dir)
    
class ParseDict(argparse.Action):
    """copied from https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d?permalink_comment_id=4134590#gistcomment-4134590"""
    def __call__(self, parser, namespace, values, option_string=None):
        d = getattr(namespace, self.dest) or {}

        if values:
            for item in values:
                split_items = item.split("=", 1)
                key = split_items[
                    0
                ].strip()  # we remove blanks around keys, as is logical
                value = split_items[1]

                d[key] = value

        setattr(namespace, self.dest, d)
        
def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('index', type=int)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single.yml')
    parser.add_argument('-ck', '--ckpt', type=str, default='trained_models/ckpt.pt')
    parser.add_argument('-i', '--index', type=int, default=None)
    parser.add_argument('-o', '--out_root', type=str, default='test_results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-tr', '--traj_idx', type=int, default=0)
    parser.add_argument(
        "--hydropathy_spec",
        metavar="KEY=VALUE",
        nargs="+",
        help="Set a number of key-value pairs "
        "(do not put spaces before or after the = sign). "
        "If a value contains spaces, you should define "
        "it with double quotes: "
        'foo="this is a sentence". Note that '
        "values are always treated as strings.",
        default=None,
        action=ParseDict,
    )
    parser.add_argument(
        "--charge_spec",
        metavar="KEY=VALUE",
        nargs="+",
        help="Set a number of key-value pairs "
        "(do not put spaces before or after the = sign). "
        "If a value contains spaces, you should define "
        "it with double quotes: "
        'foo="this is a sentence". Note that '
        "values are always treated as strings.",
        action=ParseDict,
        default=None,
    )

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    main()
