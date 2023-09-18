### Reproduce the results of CDRH3 docking
```bash
python design_testset.py --config configs/test/dock_design_single.yml  -ck reproduction/dock_single_cdr/250000.pt  -o "cdrh3_docking_results"  -d "cuda:7" -e --topk 1
```
