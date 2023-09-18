from collections import defaultdict
import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src.datasets import get_dataset
from src.models import get_model
from src.modules.common.geometry import reconstruct_backbone_partially
from src.modules.common.so3 import so3vec_to_rotation
from src.tools.eval.run import run_energy
from src.tools.relax.run import run_relax
from src.utils.inference import RemoveNative
from src.utils.protein.writers import save_pdb
from src.utils.train import recursive_to
from src.utils.misc import *
from src.utils.data import *
from src.utils.transforms import *
from src.utils.inference import *
from DockQ.DockQ import calc_DockQ
from src.utils.transforms._base import _mask_select_data
from easydict import EasyDict as edict
import pandas as pd


def create_data_variants(config, structure_factory):
    structure = structure_factory()
    structure_id = structure["id"]

    data_variants = []
    if config.mode == "single_cdr":
        cdrs = sorted(
            list(set(find_cdrs(structure)).intersection(config.sampling.cdrs))
        )
        for cdr_name in cdrs:
            transform = Compose(
                [
                    MaskSingleCDR(cdr_name, augmentation=False),
                    MergeChains(),
                ]
            )
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            data_variants.append(
                {
                    "data": data_var,
                    "name": f"{structure_id}-{cdr_name}",
                    "tag": f"{cdr_name}",
                    "cdr": cdr_name,
                    "residue_first": residue_first,
                    "residue_last": residue_last,
                }
            )
    elif config.mode == "multiple_cdrs":
        cdrs = sorted(
            list(set(find_cdrs(structure)).intersection(config.sampling.cdrs))
        )
        transform = Compose(
            [
                MaskMultipleCDRs(selection=cdrs, augmentation=False),
                MergeChains(),
            ]
        )
        data_var = transform(structure_factory())
        data_variants.append(
            {
                "data": data_var,
                "name": f"{structure_id}-MultipleCDRs",
                "tag": "MultipleCDRs",
                "cdrs": cdrs,
                "residue_first": None,
                "residue_last": None,
            }
        )
    elif config.mode == "dock_antibody":
        transform = Compose(
            [
                MaskFullAntibody(antibody_chains=config.sampling.antibody_chains),
                MergeChains(),
            ]
        )
        data_var = transform(structure_factory())
        data_variants.append(
            {
                "data": data_var,
                "name": f"{structure_id}-{config.mode}",
                "tag": config.mode,
                "residue_first": None,
                "residue_last": None,
            }
        )

    elif config.mode == "full":
        transform = Compose(
            [
                MaskAntibody(),
                MergeChains(),
            ]
        )
        data_var = transform(structure_factory())
        data_variants.append(
            {
                "data": data_var,
                "name": f"{structure_id}-Full",
                "tag": "Full",
                "residue_first": None,
                "residue_last": None,
            }
        )
    elif config.mode == "abopt":
        cdrs = sorted(
            list(set(find_cdrs(structure)).intersection(config.sampling.cdrs))
        )
        for cdr_name in cdrs:
            transform = Compose(
                [
                    MaskSingleCDR(cdr_name, augmentation=False),
                    MergeChains(),
                ]
            )
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append(
                    {
                        "data": data_var,
                        "name": f"{structure_id}-{cdr_name}-O{opt_step}",
                        "tag": f"{cdr_name}-O{opt_step}",
                        "cdr": cdr_name,
                        "opt_step": opt_step,
                        "residue_first": residue_first,
                        "residue_last": residue_last,
                    }
                )
    else:
        raise ValueError(f"Unknown mode: {config.mode}.")
    return data_variants


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "-ck",
        "--ckpt",
        type=str,
        default="logs/seq_design__antigen-20/checkpoints/300000.pt",
    )
    parser.add_argument(
        "-c", "--config", type=str, default="configs/test/seq_design.yml"
    )
    parser.add_argument("-o", "--out_root", type=str, default="seq_design_results/")
    parser.add_argument("-r", "--relax", action="store_true", default=False)
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    parser.add_argument("-t", "--tag", type=str, default="antigen-20")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument(
        "-e",
        "--eval_all",
        action="store_true",
        default=False,
        help="Call `eval_all` fn to evaluate all structures in the testset. ",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="topk for metric. must be less than or equal to num_samples.",
    )
    args = parser.parse_args()
    assert args.topk <= args.num_samples
    return args


def eval_all(args):
    """
    args contains arguments in parse_args() except for index, which is used to iterate over the testset.

    saved dataframe is like:

            DockQ   irms    Lrms   fnat  AAR
    H_CDR3  0.205  4.972  12.332  0.173  1.0
    retrun dict is like:
        - variant
            - metric
                - value
    """
    config, config_name = load_config(args.config)
    dataset = get_dataset(config.dataset.test)
    dname = os.path.join(
        args.out_root, config_name + "_%s" % args.tag if args.tag else ""
    )

    testset_results = {}
    results = []
    for i in range(len(dataset)):
        args.index = i
        result_dict = main(args)
        testset_results[i] = result_dict
        results.append(result_dict)
    testset_results = combine_nested_dicts(results)
    mean_result = traverse_dict(
        testset_results, list, lambda x: round(float(np.mean(x)), 3), lambda x: x
    )
    pd.DataFrame.from_dict(mean_result, orient="index").to_csv(
        os.path.join(dname, "testset_results.csv")
    )
    return mean_result


def main(args):
    """
    Args:
        args: arguments from parse_args()
    Returns:
        mean_results (dict):
            - variant tag, e.g., 'H_CDR3'
                - metric name, e.g., 'DockQ', 'AAR'
                    - metric value
    """

    # Load configs
    config, config_name = load_config(args.config)

    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Testset
    dataset = get_dataset(config.dataset.test)
    get_structure = lambda: dataset[args.index]

    # Logging
    structure_ = get_structure()
    structure_id = structure_["id"]
    dname = os.path.join(
        args.out_root, config_name + "_%s" % args.tag if args.tag else ""
    )
    log_dir = get_new_log_dir(dname, prefix="%04d_%s" % (args.index, structure_["id"]))
    logger = get_logger("sample", log_dir)
    logger.info("Data ID: %s" % structure_["id"])
    data_native = MergeChains()(structure_)
    save_pdb(data_native, os.path.join(log_dir, "reference.pdb"))

    # Load checkpoint and model
    logger.info("Loading model config and checkpoints: %s" % (args.ckpt))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_ckpt = ckpt["config"]
    model_cfg = cfg_ckpt.model
    model = get_model(model_cfg).to(args.device)
    lsd = model.load_state_dict(ckpt["model"])
    logger.info(str(lsd))

    # Make data variants
    data_variants = create_data_variants(
        config=config,
        structure_factory=get_structure,
    )

    # Save metadata
    metadata = {
        "identifier": structure_id,
        "index": args.index,
        "config": args.config,
        "items": [
            {kk: vv for kk, vv in var.items() if kk != "data"} for var in data_variants
        ],
    }
    with open(os.path.join(log_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Start sampling
    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [
        PatchAroundAnchor(
            initial_patch_size=model_cfg.initial_patch_size,
            antigen_size=model_cfg.antigen_size,
            remove_anchor=model_cfg.remove_anchor,
            crop_contiguous_antigen=model_cfg.get("crop_contiguous_antigen", False),
            contiguous_threshold=model_cfg.get("contiguous_threshold", 1e6),
        ),
    ]
    if "abopt" not in config.mode:  # Don't remove native CDR in optimization mode
        inference_tfm.append(
            RemoveNative(
                remove_structure=config.sampling.sample_structure,
                remove_sequence=config.sampling.sample_sequence,
            )
        )
    inference_tfm = Compose(inference_tfm)
    result_dict = {}
    aa_df = pd.DataFrame(columns=["Region", "native_aa", "sampled_aa", "AAR" , "PPL"])
    for variant in data_variants:
        variant_result_dict = defaultdict(list)
        os.makedirs(os.path.join(log_dir, variant["tag"]), exist_ok=True)
        logger.info(f"Start sampling for: {variant['tag']}")

        data_cropped = inference_tfm(copy.deepcopy(variant["data"]))
        data_list_repeat = [data_cropped] * args.num_samples
        loader = DataLoader(
            data_list_repeat,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        count = 0
        candidates = []
        for batch in tqdm(loader, desc=variant["name"], dynamic_ncols=True):
            torch.set_grad_enabled(False)
            model.eval()
            batch = recursive_to(batch, args.device)
            if "abopt" in config.mode:
                # Antibody optimization starting from native
                traj_batch = model.optimize(
                    batch,
                    opt_step=variant["opt_step"],
                    optimize_opt={
                        "pbar": True,
                        "sample_structure": config.sampling.sample_structure,
                        "sample_sequence": config.sampling.sample_sequence,
                    },
                )
            else:
                # De novo design
                traj_batch = model.sample(
                    batch,
                    sample_opt={
                        "pbar": True,
                        "sample_structure": config.sampling.sample_structure,
                        "sample_sequence": config.sampling.sample_sequence,
                        "contig":"", # do not specify contig regions on test set evaluation.
                    },
                )

            aa_new = traj_batch[0][2]  # 0: Last sampling step. 2: Amino acid.
            prmsd = traj_batch[0][3]  # 3: prmsd, [B, N]
            perplexity = traj_batch[0][4]  # 4: perplexity, [B,]
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx=batch["pos_heavyatom"],
                R_new=so3vec_to_rotation(traj_batch[0][0]),
                t_new=traj_batch[0][1],
                aa=aa_new,
                chain_nb=batch["chain_nb"],
                res_nb=batch["res_nb"],
                mask_atoms=batch["mask_heavyatom"],
                mask_recons=batch["generate_flag"],
            )
            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()
            mask_atom_new = mask_atom_new.cpu()

            # save ref docking pose
            data_tmpl = variant["data"]
            antigen_mask = data_tmpl["fragment_type"] == constants.Fragment.Antigen
            patch_mask = torch.zeros_like(antigen_mask).bool()
            patch_mask[data_cropped["patch_idx"]] = True
            antigen_and_patch_mask = antigen_mask | patch_mask
            native_patch = _mask_select_data(data_tmpl, antigen_and_patch_mask)
            ref_path = os.path.join(log_dir, variant["tag"], "REF1.pdb")
            save_pdb(native_patch, path=ref_path)

            for i in range(aa_new.size(0)):
                data_tmpl = variant["data"]
                generate_flag_i = batch["generate_flag"][i]
                prmsd_i = round(
                    float(prmsd[i].mean()), 6
                )  # round(float(prmsd[i][generate_flag_i].mean()),3)
                perplexity_i = round(float(perplexity[i]), 6)

                pred_aa = aa_new[i][generate_flag_i]
                aa = apply_patch_to_tensor(
                    data_tmpl["aa"], aa_new[i], data_cropped["patch_idx"]
                )
                mask_ha = apply_patch_to_tensor(
                    data_tmpl["mask_heavyatom"],
                    mask_atom_new[i],
                    data_cropped["patch_idx"],
                )
                pos_ha = apply_patch_to_tensor(
                    data_tmpl["pos_heavyatom"],
                    pos_atom_new[i] + batch["origin"][i].view(1, 1, 3).cpu(),
                    data_cropped["patch_idx"],
                )
                native_aa = data_tmpl["aa"][data_tmpl["generate_flag"]]
                generated_structure = pos_ha[data_tmpl["generate_flag"]]
                candidates.append(generated_structure)
                aar = float((native_aa == pred_aa).sum().float() / native_aa.size(0))
                native_aa_codes = "".join(
                    constants.aaidx2symbol[int(i)] for i in native_aa
                )
                pred_aa_codes = "".join(constants.aaidx2symbol[int(i)] for i in pred_aa)
                aa_df = pd.concat(
                    [
                        aa_df,
                        pd.DataFrame(
                            {
                                "Region": [variant["tag"]],
                                "native_aa": [native_aa_codes],
                                "sampled_aa": [pred_aa_codes],
                                "AAR": [aar],
                                "PPL": [perplexity_i],
                            }
                        ),
                    ]
                )
                # save cdr + antigen complex.
                antigen_mask = data_tmpl["fragment_type"] == constants.Fragment.Antigen
                patch_mask = torch.zeros_like(antigen_mask).bool()
                patch_mask[data_cropped["patch_idx"]] = True
                antigen_and_patch_mask = antigen_mask | patch_mask
                data = copy.deepcopy(data_tmpl)
                data["aa"] = aa
                data["mask_heavyatom"] = mask_ha
                data["pos_heavyatom"] = pos_ha
                data_patch = _mask_select_data(data, antigen_and_patch_mask)
                save_path = os.path.join(log_dir, variant["tag"], "%04d.pdb" % (count,))
                save_pdb(data_patch, path=save_path)
                del data, data_patch
                # save_pdb({
                #     'chain_nb': data_tmpl['chain_nb'],
                #     'chain_id': data_tmpl['chain_id'],
                #     'resseq': data_tmpl['resseq'],
                #     'icode': data_tmpl['icode'],
                #     # Generated
                #     'aa': aa,
                #     'mask_heavyatom': mask_ha,
                #     'pos_heavyatom': pos_ha,
                # }, path=save_path)

                patch_path = os.path.join(
                    log_dir, variant["tag"], "%04d_patch.pdb" % (count,)
                )
                save_pdb(
                    {
                        "chain_nb": data_cropped["chain_nb"],
                        "chain_id": data_cropped["chain_id"],
                        "resseq": data_cropped["resseq"],
                        "icode": data_cropped["icode"],
                        # Generated
                        "aa": aa_new[i],
                        "mask_heavyatom": mask_atom_new[i],
                        "pos_heavyatom": pos_atom_new[i]
                        + batch["origin"][i].view(1, 1, 3).cpu(),
                    },
                    path=patch_path,
                )
                count += 1

                score_dict = calc_DockQ(save_path, ref_path, use_CA_only=True)
                score_dict = {
                    k: round(v, 3)
                    for k, v in score_dict.items()
                    if k in ["DockQ", "irms", "Lrms", "fnat"]
                }
                score_dict.update({"AAR": aar, "prmsd": prmsd_i})
                [variant_result_dict[key].append(score_dict[key]) for key in score_dict]
            logger.info("Finished.\n")
        candidates = torch.stack(candidates, dim=0)  # (n_candidates, n_res, n_atoms, 3)
        candidates = candidates[
            ...,
            (
                constants.BBHeavyAtom.N,
                constants.BBHeavyAtom.CA,
                constants.BBHeavyAtom.CB,
            ),
            :,
        ]
        candidates = candidates.reshape(count, -1, 3)
        topk_idxs = rank_commoness(candidates, args.topk)
        top_result = {
            f"{k}_top{args.topk}": [v[i] for i in topk_idxs]
            for k, v in variant_result_dict.items()
        }
        variant_result_dict.update(top_result)
        result_dict[variant["tag"]] = variant_result_dict
    results = traverse_dict(
        result_dict, list, lambda x: round(float(np.mean(x)), 3), lambda x: f"{x}_mean"
    )
    # mean_results = traverse_dict(result_dict, list, lambda x: round(float(np.mean(x)),3), lambda x: x)
    topk_results = extract_dict(
        result_dict, list, lambda k: k.endswith(f"_top{args.topk}")
    )
    df = pd.DataFrame.from_dict(results, orient="index")
    df.to_csv(os.path.join(log_dir, "results.csv"))
    aa_df.to_csv(os.path.join(log_dir, "aa.csv"), index=False)
    if args.relax:
        run_relax(log_dir, pipeline_name="openmm_pyrosetta")
        run_energy(log_dir)
        openmm_path = save_path.replace(".pdb", "_openmm.pdb")
        rosetta_path = save_path.replace(".pdb", "_rosetta.pdb")
    return topk_results  # mean_results


def traverse_dict(d, leaf_type, leaf_fn, leaf_key_fn, leaf_key_vali_fn=lambda x: True):
    """
    traverse a dict and apply a function to all leaves if they are of a certain type
    return a new dict with the same structure as the input dict, but with the leaves replaced by the result of the function.
    :param d: dict to traverse
    :param leaf_type: type of the leaves to apply the function to
    :param leaf_fn: function to apply to the leaves
    :return: new dict with the same structure as the input dict, but with the leaves replaced by the result of the function.
    """
    ret_dict = copy.deepcopy(d)
    for k, v in d.items():
        if isinstance(v, dict):
            ret_dict[k] = traverse_dict(v, leaf_type, leaf_fn, leaf_key_fn)
        elif isinstance(v, leaf_type) and leaf_key_vali_fn(k):
            ret_dict[leaf_key_fn(k)] = leaf_fn(v)
    return ret_dict


def extract_dict(d, leaf_type, leaf_key_vali_fn=lambda x: True):
    """
    extract a dict with only leaves of a certain type
    return a new dict with the same structure as the input dict, but with the leaves replaced by the result of the function.
    :param d: dict to traverse
    :param leaf_type: type of the leaves to apply the function to
    :param leaf_fn: function to apply to the leaves
    :return: new dict with the same structure as the input dict, but with the leaves replaced by the result of the function.
    """
    ret_dict = copy.deepcopy(d)
    for k, v in d.items():
        if isinstance(v, dict):
            ret_dict[k] = extract_dict(v, leaf_type, leaf_key_vali_fn)
        elif isinstance(v, leaf_type) and leaf_key_vali_fn(k):
            ret_dict[k] = v
        else:
            del ret_dict[k]
    return ret_dict


def combine_nested_dicts(dicts):
    """
    Combine a list of nested dictionaries into a single nested dictionary.

    Args:
        dicts (list): List of nested dictionaries with similar keys.

    Returns:
        dict: A nested dictionary with the same structure as the input dictionaries.
    """
    if len(dicts) == 1:
        return dicts[0]
    else:
        combined_dict = {}
        for key in dicts[0].keys():
            if isinstance(dicts[0][key], dict):
                combined_dict[key] = combine_nested_dicts([d[key] for d in dicts])
            else:
                combined_dict[key] = [d[key] for d in dicts]
        return combined_dict


def calc_per_rmsd(structures):
    B, N, _ = structures.shape
    structures = structures.unsqueeze(1).repeat(1, B, 1, 1)  # (B, B, N, 3)
    structures_ = structures.permute(1, 0, 2, 3)  # (B, B, N, 3)
    rmsd = torch.sqrt(
        (((structures - structures_) ** 2).sum(dim=-1)).mean(-1)
    )  # (B, B)
    return rmsd


def calc_avg_rmsd(structures):
    B, N, _ = structures.shape
    rmsd = calc_per_rmsd(structures)
    avg_rmsd = rmsd.sum() / (B * (B - 1))
    return avg_rmsd


def rank_commoness(structures, k):
    """
    Find out the most common structure within a list of structures.
    A structure is common if it is similar to all other structures.
    The similarity is measured by RMSD. The lower the RMSD, the more similar the structures are.
    Args:
        structures(torch.Tensor): (B, N, 3). B is num of structures, N is the structure length.
    Returns:
        rank (torch.Tensor): (B, ), the similarity rank of each structure.
    """
    B, N, _ = structures.shape
    rmsd = calc_per_rmsd(structures)  # (B, B)
    similarity, rank = torch.topk(
        rmsd.sum(dim=-1) / (B - 1), k=k, largest=False
    )  # (B, )

    return rank
