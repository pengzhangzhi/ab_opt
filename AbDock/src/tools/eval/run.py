import os
import argparse
import ray
import shelve
import time
import pandas as pd
from typing import Mapping

from src.tools.eval.base import EvalTask, TaskScanner
from src.tools.eval.similarity import eval_similarity
from src.tools.eval.energy import eval_interface_energy


@ray.remote(num_cpus=1)
def evaluate(task, args):
    return evaluate_local(task, args.no_energy)


def evaluate_local(task, no_energy):
    funcs = []
    funcs.append(eval_similarity)
    if not no_energy:
        funcs.append(eval_interface_energy)
    for f in funcs:
        task = f(task)
    return task

def dump_db(db: Mapping[str, EvalTask], path):
    table = []
    for task in db.values():
        if 'abopt' in path and task.scores['seqid'] >= 100.0:
            # In abopt (Antibody Optimization) mode, ignore sequences identical to the wild-type
            continue
        table.append(task.to_report_dict())
    table = pd.DataFrame(table)
    table.to_csv(path, index=False, float_format='%.6f')
    return table

def run_energy(root, pfx='rosetta', no_energy=False):
    db_path = os.path.join(root, 'evaluation_db')
    with shelve.open(db_path) as db:
        scanner = TaskScanner(root=root, postfix=pfx, db=db)
        tasks = scanner.scan()
        for task in tasks:
            done_task = evaluate_local(task, no_energy)
            done_task.save_to_db(db)
        db.sync()
        dump_db(db, os.path.join(root, 'summary.csv'))
        
def main(args):
    
    db_path = os.path.join(args.root, 'evaluation_db')
    with shelve.open(db_path) as db:
        scanner = TaskScanner(root=args.root, postfix=args.pfx, db=db)

        tasks = scanner.scan()
        futures = [evaluate.remote(t, args) for t in tasks]
        if len(futures) > 0:
            print(f'Submitted {len(futures)} tasks.')
        ray.get(futures)
        # while len(futures) > 0: 
            # done_ids, futures = ray.wait(futures, num_returns=1)
            # for done_id in done_ids:
            #     done_task = ray.get(done_id)
            #     done_task.save_to_db(db)
            #     print(f'Remaining {len(futures)}. Finished {done_task.in_path}')
        db.sync()
        
        dump_db(db, os.path.join(args.root, 'summary.csv'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--no_energy', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ray.init()
    args = parse_args()
    main(args)
