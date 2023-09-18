from collections import defaultdict
import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from DockQ.DockQ import calc_DockQ
# from src.tools.relax.run import run_relax
from src.utils.transforms._base import _mask_select_data
import pandas as pd
from src.datasets.custom import preprocess_antibody_structure
from src.models import get_model
from src.modules.common.geometry import reconstruct_backbone_partially
from src.modules.common.so3 import so3vec_to_rotation
from src.tools.runner.design_for_testset import (
    create_data_variants,
    extract_dict,
    rank_commoness,
    traverse_dict,
    calc_avg_rmsd,
)
from src.utils.inference import RemoveNative
from src.utils.protein.writers import save_pdb
from src.utils.train import recursive_to
from src.utils.misc import *
from src.utils.data import *
from src.utils.transforms import *
from src.utils.inference import *
from src.tools.renumber import renumber as renumber_antibody


def dock_for_pdb(args):
    # Load configs
    config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Structure loading
    data_id = os.path.basename(args.pdb_path) if args.id == '' else args.id
    if args.label_heavy_as_cdr:
        label_whole_heavy_chain_as_cdr = True
        pdb_path = args.pdb_path
        assert (
            args.heavy is not None
        ), f"must specify heavy chain id for seq designed pdb."
    else:
        label_whole_heavy_chain_as_cdr = False
        if args.no_renumber:
            pdb_path = args.pdb_path
        else:
            in_pdb_path = args.pdb_path
            out_pdb_path = os.path.splitext(in_pdb_path)[0] + "_chothia.pdb"
            heavy_chains, light_chains = renumber_antibody(in_pdb_path, out_pdb_path)
            pdb_path = out_pdb_path

            if args.heavy is None and len(heavy_chains) > 0:
                args.heavy = heavy_chains[0]
            if args.light is None and len(light_chains) > 0:
                args.light = light_chains[0]
        if args.heavy is None and args.light is None:
            raise ValueError(
                "Neither heavy chain id (--heavy) or light chain id (--light) is specified."
            )
    get_structure = lambda: preprocess_antibody_structure(
        {
            "id": data_id,
            "pdb_path": pdb_path,
            "heavy_id": args.heavy,
            # If the input is a nanobody, the light chain will be ignores
            "light_id": args.light,
        },
        label_whole_heavy_chain_as_cdr=label_whole_heavy_chain_as_cdr,
    )

    # Logging
    structure_ = get_structure()
    structure_id = structure_["id"]
    tag_postfix = "_%s" % args.tag if args.tag else ""
    log_dir = get_new_log_dir(
        os.path.join(args.out_root, config_name + tag_postfix), prefix=data_id
    )
    logger = get_logger("sample", log_dir)
    logger.info("Data ID: %s" % structure_["id"])
    logger.info(f"Results will be saved to {log_dir}")
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
    if "abopt" not in config.mode and args.contig == '':  # Don't remove native CDR in optimization mode
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
            traj_batch = model.sample(
                batch,
                sample_opt={
                    "pbar": True,
                    "sample_structure": config.sampling.sample_structure,
                    "sample_sequence": config.sampling.sample_sequence,
                    "contig": args.contig,
                },
            )
            batch = recursive_to(batch, 'cpu')
            traj_batch = recursive_to(traj_batch, 'cpu')
            def save_traj(traj_batch, save_path):
                pdb_str = ""
                for i in reversed(range(len(traj_batch))):
                    aa_new_i = traj_batch[i][2]  
                    pos_atom_new_i, mask_atom_new_i = reconstruct_backbone_partially(
                        pos_ctx=batch["pos_heavyatom"],
                        R_new=so3vec_to_rotation(traj_batch[i][0]),
                        t_new=traj_batch[i][1],
                        aa=aa_new_i,
                        chain_nb=batch["chain_nb"],
                        res_nb=batch["res_nb"],
                        mask_atoms=batch["mask_heavyatom"],
                        mask_recons=batch["generate_flag"],
                    )   
                    aa_new_i = aa_new_i.cpu()[0]
                    pos_atom_new_i = pos_atom_new_i.cpu()[0]
                    mask_atom_new_i = mask_atom_new_i.cpu()[0]
                    data_tmpl = variant["data"]
                    antigen_mask = data_tmpl["fragment_type"] == constants.Fragment.Antigen
                    patch_mask = torch.zeros_like(antigen_mask).bool()
                    patch_mask[data_cropped["patch_idx"]] = True
                    antigen_and_patch_mask = antigen_mask | patch_mask
                    data = copy.deepcopy(data_tmpl)
                    mask_ha = apply_patch_to_tensor(
                        data_tmpl["mask_heavyatom"],
                        mask_atom_new_i,
                        data_cropped["patch_idx"],
                    )
                    pos_ha = apply_patch_to_tensor(
                        data_tmpl["pos_heavyatom"],
                        pos_atom_new_i + batch["origin"][0].view(1, 1, 3).cpu(),
                        data_cropped["patch_idx"],
                    )
                    data["mask_heavyatom"] = mask_ha
                    data["pos_heavyatom"] = pos_ha
                    data_patch = _mask_select_data(data, antigen_and_patch_mask)
                    save_pdb(data_patch, path='tmp.pdb')
                    with open('tmp.pdb', 'r') as f:
                        lines = f.read().splitlines()
                        lines = [f'MODEL     {i+1}'] + lines + ['ENDMDL\n']
                        pdb_str += '\n'.join(lines)
                with open(save_path, 'w') as f:
                    f.write(pdb_str)
            save_traj(traj_batch, os.path.join(log_dir, "traj.pdb"))
            
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
                generate_flag_i = batch["generate_flag"][i].cpu()
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
                gen_flag_full = apply_patch_to_tensor(
                    data_tmpl["generate_flag"], generate_flag_i, data_cropped["patch_idx"]
                )
                native_aa = data_tmpl["aa"][gen_flag_full]
                generated_structure = pos_ha[gen_flag_full]
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
                score_dict = {}
                dock_dict = {
                    k: round(v, 3)
                    for k, v in calc_DockQ(save_path, ref_path, use_CA_only=True).items()
                    if k in ["DockQ", "irms", "Lrms", "fnat"]
                }
                score_dict.update(dock_dict)
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
        avg_rmsd = float(calc_avg_rmsd(candidates))
        top_result = {
            f"{k}_top{args.topk}": [v[i] for i in topk_idxs]
            for k, v in variant_result_dict.items() if isinstance(v, list)
        }
        variant_result_dict.update({"per_sample_rmsd": avg_rmsd})
        
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

    return topk_results  # mean_results


def args_from_cmdline():
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="", help="name of the pdb, use for saved dir. if not specified, use the pdb name.")
    parser.add_argument(
        "--pdb_path",
        type=str,
        default="data/examples/7DK2_AB_C.pdb",
        help="Path to the PDB file.",
    )
    parser.add_argument(
        "--label_heavy_as_cdr",
        default=False,
        action="store_true",
        help="Label the heavy chain as CDR. \
            If True, specify the heavy chain id with --heavy. \
        Usually used when the heavy chain contains only designed CDRH3.",
    )
    parser.add_argument(
        "--contig",
        default='',
        help="Specify contiguous residues to be designed. \
            e.g., 1-10, meaning the first to tenth CDR residues will be designed. \
            If not specified, all residues will be designed.",
    )

    parser.add_argument(
        "-c", "--config", type=str, default="configs/test/dock_cdr.yml"
    )
    parser.add_argument(
        "-ck", "--ckpt", type=str, default="reproduction/dock_single_cdr/250000.pt"
    )
    parser.add_argument(
        "--heavy", type=str, default=None, help="Chain id of the heavy chain."
    )
    parser.add_argument(
        "--light", type=str, default=None, help="Chain id of the light chain."
    )
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Select topk samples. must be less than or equal to num_samples.",
    )
    parser.add_argument("--no_renumber", action="store_true", default=False)
    parser.add_argument("-o", "--out_root", type=str, default="./results/")
    parser.add_argument("-t", "--tag", type=str, default="")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    args = parser.parse_args()
    assert args.topk <= args.num_samples
    return args


def args_factory(**kwargs):
    default_args = EasyDict(
        heavy="H",
        light="L",
        no_renumber=False,
        config="./configs/test/codesign_single.yml",
        out_root="./results",
        tag="",
        seed=None,
        device="cuda",
        batch_size=16,
    )
    default_args.update(kwargs)
    return default_args


if __name__ == "__main__":
    dock_for_pdb(args_from_cmdline())
