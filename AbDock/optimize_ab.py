from easydict import EasyDict
import numpy as np
import os
import pandas as pd
import glob
import subprocess
import ray
import argparse

from src.tools.relax.run import main as relax_main
from src.tools.eval.run import main as eval_main
from src.tools.relax.run import pipeline_pyrosetta_fixbb,pipeline_openmm_pyrosetta

def seq_design_batch(save_dir, pose_dir, nums, heavy_chain_id, checkpoint_path, contig='',pdb_suffix='rosetta'):
    """
    run seq design procedure for sampled docking poses.
    """
    # initialize ray with the number of GPUs to use

    # specify the number of GPUs to use for each remote function call
    @ray.remote(num_gpus=1/args.process_per_gpu, num_cpus=1)
    def dock_pdb(path: str, save_dir: str, nums: int, heavy_chain_id: str):
        print(f"Processing pdb {path}")
        pdb_id = path.split('/')[-1].split('_')[0]
        cmd = f'python dock_pdb.py --pdb_path {path} -c configs/test/seq_design.yml\
            -ck {checkpoint_path} -o {save_dir} -n {nums} -b {nums} --heavy {heavy_chain_id} \
                --label_heavy_as_cdr'
        cmd += f' --contig {contig}' if contig else ''
        print(cmd)
        os.system(cmd)
        return path
    # create a list of remote function calls
    remote_calls = []
    for path in glob.glob(pose_dir+f'/*_{pdb_suffix}.pdb'):
        print(f"Submitting job for pdb {path}")
        remote_calls.append(dock_pdb.remote(path, save_dir, nums, heavy_chain_id))
    
    # retrieve the results from the remote function calls
    ray.get(remote_calls)
    summarize_seqs(save_dir)
        
def summarize_seqs(design_dir):
    out_df = pd.DataFrame(columns=['pdb_id', 'AAR'])
    
    for path in glob.glob(design_dir+'/*/aa.csv'):
        pdb_id = path.split('/')[-2]
        df = pd.read_csv(path)
        aar = df['AAR'].mean()
        out_df = out_df.append({'pdb_id': pdb_id, 'AAR': aar}, ignore_index=True) 
    out_df.to_csv(design_dir+'/summary.csv', index=False)
    return out_df

def gen_poses(native_path, out_dir, nums, checkpoint_path):
    cmd = f'python dock_pdb.py --pdb_path {native_path} -c configs/test/dock_cdr.yml\
                -ck {checkpoint_path} -o {out_dir} -n {nums} -b {nums}\
                   '
    print(cmd)
    os.system(cmd)

def dock_seqs(design_dir, out_dir, nums, heavy_chain_id, checkpoint_path, pdb_suffix):
    """
    Dock the designed sequences to the antigen.
    cmd:
    py dock_pdb.py --pdb_path wet_experiments/opt_ab/results/seq_design/7bsd/seq_design/0117_patch.pdb_/H_CDR3/0000_patch.pdb -c configs/test/dock_design_single.yml -ck reproduction/dock_single_cdr/250000.pt -o wet_experiments/opt_ab/results/screening -n 100 -b 100 --heavy A --label_heavy_as_cdr
    """
    @ray.remote(num_gpus=1/args.process_per_gpu, num_cpus=1)
    def dock_seq(path, id):
        cmd = f'python dock_pdb.py --id {id} --pdb_path {path} -c configs/test/dock_cdr.yml\
                -ck {checkpoint_path} -o {out_dir} -n {nums} -b {nums}\
                    --heavy {heavy_chain_id} --label_heavy_as_cdr'
        print(cmd)
        os.system(cmd)

    
    tasks = []
    paths = glob.glob(design_dir+f'/*_{pdb_suffix}.pdb_/H_CDR3/0000.pdb')
    if len(paths) == 0:
        raise ValueError(f'No pdb found in {design_dir}')
    for path in paths:
        id = path.split('/')[-3]
        print(id)
        tasks.append(dock_seq.remote(path, id))
        
    ray.get(tasks)
    ray.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script description')    
    
    parser.add_argument('--num_gpus', type=int, default=7, help='Number of GPUs')
    parser.add_argument('--process_per_gpu', type=int, default=1, help='Number of processes per GPU')
    parser.add_argument('--docked_pose_dir', type=str, default='wet_experiments/opt_ab/co_optimization/results/dock_single/7bsd_A_B_G.pdb_/H_CDR3', help='Docked pose directory')
    parser.add_argument('--seq_design_dir', type=str, default='wet_experiments/opt_ab/co_optimization/results/seq_design_fixed_pos/mutation/CDRH3_7_9', help='Sequence design directory')
    parser.add_argument('--design_model_ckpt', type=str, default='reproduction/seq_design_fixed_pos/300000.pt', help='Design model checkpoint')
    parser.add_argument('--design_contig', type=str, default='', help='Design contig')
    parser.add_argument('--screen_dir', type=str, default='wet_experiments/opt_ab/co_optimization/results/screening/seq_design_fixed_pos/mutation/CDRH3_7_9', help='Screening directory')
    parser.add_argument('--dock_model_ckpt', type=str, default='reproduction/dock_single_cdr/250000.pt', help='Dock model checkpoint')
    parser.add_argument('--heavy_chain_id', type=str, default='A', help='Heavy chain ID')
    parser.add_argument('--nums', type=int, default=100, help='Number of sequences')
    parser.add_argument('--pdb_suffix', type=str, default='rosetta', help='PDB suffix')

    args = parser.parse_args()
    os.makedirs(args.seq_design_dir, exist_ok=True)
    os.makedirs(args.screen_dir, exist_ok=True)
    
    ray.init(num_gpus=args.num_gpus)
    
    
    # run rosetta relax
    relax_main(EasyDict({
            'root': os.path.dirname(args.docked_pose_dir),
            "pipeline":pipeline_openmm_pyrosetta
    }))
    # run rosetta ddg estimation
    eval_main(args=EasyDict({
        'root': os.path.dirname(args.docked_pose_dir),
        'pfx': 'rosetta',
        'no_energy':False,
    }))

    
    seq_design_batch(
        save_dir=args.seq_design_dir,
        pose_dir=args.docked_pose_dir,
        nums=args.nums,
        heavy_chain_id=args.heavy_chain_id,
        checkpoint_path=args.design_model_ckpt,
        contig=args.design_contig,
        pdb_suffix=args.pdb_suffix,
    )

    dock_seqs(
        f'{args.seq_design_dir}/seq_design',
        args.screen_dir,
        nums=args.nums,
        heavy_chain_id=args.heavy_chain_id,
        checkpoint_path=args.dock_model_ckpt,
        pdb_suffix=args.pdb_suffix,
    )

    