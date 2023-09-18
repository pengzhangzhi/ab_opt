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

```bash
python design_testset.py --hydropathy_spec 1=+ --charge_spec 2=-
``` 
- The syntax is `position=label`. where `1` and `2` are the positions of the amino acid in the sequence, and `+` and `-` are the labels. The symbols are defined as follows. 
```python
char2hydropathy = {
    '+': Hydropathy.hydrophilic,
    '-': Hydropathy.moderate,
    '?': Hydropathy.unknown,
}
char2charge = {
    '+': Charge.positive,
    '-': Charge.negtive,
    '=': Charge.neutral,
    '?': Charge.unknown,
}
```
- Multiple labels can be specified by separating them with space, e.g., `python design_testset.py --hydropathy_spec 1=+ 2=- --charge_spec 3=+ 4=-`


## Train

```bash
python train.py --config ./configs/train/aa_label.yml
```


## Trained Weights
Trained model weights are available at `ckpt.pt`