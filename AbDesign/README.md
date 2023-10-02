# AbDesign
Code implementation of AbDesign, a generative diffusion model for antibody sequence and structure co-design.
## Install

```bash
conda env create -f env.yaml 
conda activate antibody_design
```

## Download and preprocess datasets 

Protein structures in the `SAbDab` dataset can be downloaded [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/). Extract `all_structures.zip` into the `data` folder. 

The `data` folder contains a snapshot of the dataset index (`sabdab_summary_all.tsv`). You may replace the index with the latest version [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/).

## Evaluation
<!-- To design the CDRH3 of an given antibody, run
```bash
python design_pdb.py
``` -->
To reproduce the design results on test set, run
```bash
python design_testset.py
```

- Please refer to `configs/test` for the configuration files of the test set design.
- Please refer to `design_pdb.py` and `design_testset.py` for the design code.
Results are saved in `./test_results` folder. 
We provide additional functionality to specify the amino acid properties of the designed sequences.
Specify amino acid labels such as hydrophobicity and charge, run

## Train

```bash
python train.py --config ./configs/train/aa_label.yml
```


## Trained Weights
Trained model weights are available at `ckpt.pt`
