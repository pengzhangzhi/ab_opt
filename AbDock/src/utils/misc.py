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
import dynamic_yaml
import pdbfixer
import openmm
ENERGY = openmm.unit.kilocalories_per_mole
LENGTH = openmm.unit.angstroms

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
    fn = "" # time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir, exist_ok=True)
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
        config = EasyDict(dynamic_yaml.load(f))
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



def openmm_relax(pdb_file, stiffness=10., tolerance=2.39, use_gpu=False):
    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    force_field = openmm.app.ForceField("amber14/protein.ff14SB.xml")
    modeller = openmm.app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    system = force_field.createSystem(modeller.topology)

    if stiffness > 0:
        stiffness = stiffness * ENERGY / (LENGTH**2)
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        for residue in modeller.topology.residues():
            for atom in residue.atoms():
                if atom.name in ["N", "CA", "C", "CB"]:
                    force.addParticle(
                            atom.index, modeller.positions[atom.index]
                    )
        system.addForce(force)

    tolerance = tolerance * ENERGY
    integrator = openmm.LangevinIntegrator(0, 0.01, 1.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")

    simulation = openmm.app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getKineticEnergy() + state.getPotentialEnergy()

    with open(pdb_file, "w") as f:
        openmm.app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f,
            keepIds=True
        )
    return energy

