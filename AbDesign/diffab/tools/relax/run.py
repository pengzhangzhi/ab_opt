import argparse
import ray
import time

from diffab.tools.relax.openmm_relaxer import run_openmm
from diffab.tools.relax.pyrosetta_relaxer import run_pyrosetta, run_pyrosetta_fixbb
from diffab.tools.relax.base import TaskScanner


@ray.remote(num_gpus=1/8, num_cpus=1)
def run_openmm_remote(task):
    return run_openmm(task)

def _run_openmm(task):
    return run_openmm(task)

@ray.remote(num_cpus=1)
def run_pyrosetta_remote(task):
    return run_pyrosetta(task)

def _run_pyrosetta(task):
    return run_pyrosetta(task)


@ray.remote(num_cpus=1)
def run_pyrosetta_fixbb_remote(task):
    return run_pyrosetta_fixbb(task)

def _run_pyrosetta_fixbb(task):
    return run_pyrosetta_fixbb(task)

@ray.remote
def pipeline_openmm_pyrosetta(task):
    funcs = [
        run_openmm_remote,
        run_pyrosetta_remote,
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)

def _pipeline_openmm_pyrosetta(task):
    funcs = [
        _run_openmm,
        _run_pyrosetta,
    ]
    for fn in funcs:
        task = fn(task)
    return task

@ray.remote
def pipeline_pyrosetta(task):
    funcs = [
        run_pyrosetta_remote,
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)

def _pipeline_pyrosetta(task):
    funcs = [
        _run_pyrosetta,
    ]
    for fn in funcs:
        task = fn(task)
    return task

@ray.remote
def pipeline_pyrosetta_fixbb(task):
    funcs = [
        run_pyrosetta_fixbb_remote,
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)

def _pipeline_pyrosetta_fixbb(task):
    funcs = [
        _run_pyrosetta_fixbb,
    ]
    for fn in funcs:
        task = fn(task)
    return task


pipeline_dict = {
    'openmm_pyrosetta': pipeline_openmm_pyrosetta,
    'pyrosetta': pipeline_pyrosetta,
    'pyrosetta_fixbb': pipeline_pyrosetta_fixbb,
}

pipeline_dict_local = {
    'openmm_pyrosetta': _pipeline_openmm_pyrosetta,
    'pyrosetta': _pipeline_pyrosetta,
    'pyrosetta_fixbb': _pipeline_pyrosetta_fixbb,
}
def run_relax(root, pipeline_name='openmm_pyrosetta'):
    assert pipeline_name in pipeline_dict_local.keys()
    pipeline = pipeline_dict_local[pipeline_name]
    final_pfx = 'fixbb' if pipeline == pipeline_pyrosetta_fixbb else 'rosetta'
    scanner = TaskScanner(root, final_postfix=final_pfx)
    tasks = scanner.scan()
    for t in tasks:
        pipeline(t)
        

def main():
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pipeline', type=lambda s: pipeline_dict[s], default=pipeline_openmm_pyrosetta)
    args = parser.parse_args()

    final_pfx = 'fixbb' if args.pipeline == pipeline_pyrosetta_fixbb else 'rosetta'
    scanner = TaskScanner(args.root, final_postfix=final_pfx)
    while True:
        tasks = scanner.scan()
        futures = [args.pipeline.remote(t) for t in tasks]
        if len(futures) > 0:
            print(f'Submitted {len(futures)} tasks.')
        while len(futures) > 0:
            done_ids, futures = ray.wait(futures, num_returns=1)
            for done_id in done_ids:
                done_task = ray.get(done_id)
                print(f'Remaining {len(futures)}. Finished {done_task.current_path}')
        time.sleep(1.0)

if __name__ == '__main__':
    main()
