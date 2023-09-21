# Pratope-epitope Docking using AbDock

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Credits](#credits)

## About <a name = "about"></a>
This is the code implementation of [AbDock](dock), a generative diffusion model-based paratope-epitope docking method.  We also integrate the code of the [antibody optimization pipeline](#opt) for reproducing the optimization results in the paper.
## Getting Started <a name = "getting_started"></a>

### Installing
The code relies on PyTorch with cuda support. We recommend using conda to install the dependencies. 
```
conda create -n antibody_dock python=3.8 -y
conda activate antibody_dock
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch -yconda install pyg::pytorch-scatter=2.0.9 -y
conda install pyyaml mmseqs2  -y 
conda install -c conda-forge pdbfixer -y
conda install -c bioconda abnumber -y
pip install -U "ray[default]"
pip install dynamic-yaml tqdm biopython pandas joblib lmdb easydict scipy wandb notebook seaborn
```
##### Install pyrosetta. 
Please obtain the username and password on the [PyRosetta website](https://www.pyrosetta.org/home).
```
wget --user={user} --password={psw} https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python38.linux/PyRosetta4.Release.python38.linux.release-350.tar.bz2
tar -xvjf PyRosetta4.Release.python38.linux.release-350.tar.bz2
cd PyRosetta4.Release.python38.linux.release-350
python setup/setup.py install
cd ..
```
## Usage <a name = "usage"></a>

### Paratope-epitope Docking using AbDock <a name = "dock"></a>

To perform docking for pdb_file `data/examples/7DK2_AB_C.pdb` and save the results to `./results/dock_cdr/7DK2_AB_C.pdb_`.


```bash
python dock_pdb.py --pdb_path data/examples/7DK2_AB_C.pdb
```

- `7DK2_AB_C.pdb` has the A (heavy chain) and B (light chain) chains of the antibody and the C chain (epitope) of the antigen.
- The script will automatically identify the CDRH3 and perform docking to the antigen.
- Please refer to `src/tools/runner/design_for_pdb.py:args_from_cmdline` for complete arguments.
- Please check out `./results/dock_cdr/7DK2_AB_C.pdb_` for the docking results.

### Antibody Optimization Pipeline <a name = "opt"></a>

The optimization pipeline consists of three steps: 
1. Pose generation using AbDock. 
2. Desgin sequences for generated poses.
3. Sequence screening using AbDock.

#### Generate four point-mutation results for 7bsd_A_B_G
Run AbDock to generate docking poses from native CDRH3 sequence.
```bash
python dock_pdb.py --pdb_path data/examples/7bsd_A_B_G.pdb --config configs/test/dock_cdr.yml --ckpt reproduction/dock_single_cdr/250000.pt -n 1000 -b 1000 -d "cuda" -o results
```

Run the sequence design and screening pipeline. The four points are specified by `--design_contig "6-9"`.
```bash
python optimize_ab.py --num_gpus 1 --process_per_gpu 1 --docked_pose_dir "results/dock_cdr/7bsd_A_B_G.pdb_/H_CDR3" --seq_design_dir "results/seq_design_fixed_pos/mutation/CDRH3_7_9" --design_model_ckpt "reproduction/seq_design_fixed_pos/300000.pt" --design_contig "6-9" --screen_dir "results/screening/seq_design_fixed_pos/mutation/CDRH3_7_9" --dock_model_ckpt "reproduction/dock_single_cdr/250000.pt" --heavy_chain_id "A" --nums 100 --pdb_suffix "rosetta"
```

Run the notebook `ab_opt_analysis_4mutations.ipynb` to analyse the optimiaztion results.

### Reproduce Training

#### Download and preprocess the dataset

Download Protein structures in the [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/). Extract all_structures.zip into the data folder.
### Launch training

```bash
python train.py configs/train/dock_single.yml
```
- Please refer to `train.py` for complete arguments. 
- Please check out `configs/train` for the list of training setting.
