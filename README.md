## Generative Diffusion Models for Antibody Design, Docking, and Optimization

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

**| [Code](https://github.com/pengzhangzhi/ab_opt) | [Paper](https://www.biorxiv.org/content/10.1101/2023.09.25.559190v1) | [Homepage](https://pengzhangzhi.github.io/ab_opt_homepage/) |**


Official implementation of [Generative Diffusion Models for Antibody Design, Docking, and Optimization](https://www.biorxiv.org/content/10.1101/2023.09.25.559190v1).

The wet experiment data used to validate our pipeline is available at [wet_experiment_data.zip](./wet_experiment_data.zip)


![Cover Image](cover.png)


## Quick start
### Installation
```bash
git clone git@github.com:pengzhangzhi/ab_opt.git
```
### AbDesign
Please take a look at the [AbDesign](./AbDesign/) on reproducing the training and evaluation of the AbDesign.
### AbDock

Please refer to the [AbDock](./AbDock/) on how to reproduce the training and evaluation of the AbDock and the antibody optimization pipeline.

## Credits <a name = "credits"></a>
This codebase is based on the following repositories, we thank the authors for their great work.
- [DockQ](https://github.com/bjornwallner/DockQ)
- [diffab](https://github.com/luost26/diffab)
- [Abnumber](https://github.com/prihoda/AbNumber)
- [PyRosetta](https://www.pyrosetta.org/)
- ...

