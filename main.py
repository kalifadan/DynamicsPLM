from __future__ import annotations
import requests
import os
import lmdb
import json
import torch
import os
import numpy as np
from pathlib import Path
import math
import pandas as pd
import re

# If not wanting to generate conformations or creating ablations (such the BioEmu ablation) - comments these lines
from rocketshp import RocketSHP, load_sequence, load_structure
from biotite.structure.io import pdb
from biotite.structure import to_sequence

from bioemu.sample import main as sample
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from utils.foldseek_util import get_struc_seq
from collections import Counter

THREEDI_ALPHABET = list("ABCDEFGHIJKLMNOPQRST")
THREEDI_INDEX = {c: i for i, c in enumerate(THREEDI_ALPHABET)}


def read_from_lmdb(filename):
    ii = 0
    error = 0
    with lmdb.open(filename) as env:
        with env.begin() as txn:
            for key, value in txn.cursor():
                data = json.loads(value.decode('utf-8'))
                if isinstance(data, int) or isinstance(value, int):
                    print(f"key - {key}, value - {value}")
                else:
                    if "shp" not in data:
                        print(f"error while reading protein")
                        error += 1
                ii += 1
    print("Data size:", ii - 1, " - errors:", error)


def download_alphafold_pdb_v4(uniprot_id, save_path):
    # Construct the URL for AlphaFold v4 PDB file using the UniProt ID
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

    # Send a GET request to fetch the PDB file
    response = requests.get(url)

    # Check if the response is successful
    if response.status_code == 200:
        # Save the PDB file to the specified path
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded AlphaFold PDB for {uniprot_id} to {save_path}")
    else:
        print(f"Failed to download AlphaFold PDB for {uniprot_id}. Status code: {response.status_code}")
        raise Exception


def save_lmdb(data_entries, save_lmdb_path):
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        else:
            return obj
    env = lmdb.open(save_lmdb_path, map_size=1 << 40)  # ~1 TB max size
    with env.begin(write=True) as txn:
        for i, entry in enumerate(data_entries):
            # Convert all tensors to lists recursively
            clean_entry = convert(entry)
            txn.put(str(i).encode(), json.dumps(clean_entry).encode())
        txn.put(b"length", str(len(data_entries)).encode())


def create_dataset_from_lmdb(path):
    device = torch.device("cuda:0")
    model = RocketSHP.load_from_checkpoint("v1", strict=False).to(device)

    entries = []
    with lmdb.open(path, readonly=True, lock=False) as env:
        with env.begin() as txn:
            for key, value in txn.cursor():
                if key == b"length":
                    continue
                try:
                    data = json.loads(value.decode("utf-8"))
                    if isinstance(data, int) or isinstance(value, int):
                        continue

                    uniprot_id = data.get("name")
                    uniprot_id = uniprot_id if "-" not in uniprot_id else uniprot_id.split("-")[1]

                    if uniprot_id:
                        save_path = f"example/{uniprot_id}.pdb"
                        if not os.path.exists(save_path):
                            download_alphafold_pdb_v4(uniprot_id, save_path)

                        # Load structure file (PDB)
                        structure = pdb.PDBFile.read(save_path).get_structure()
                        struct_features = load_structure(structure, device=device)

                        # Get sequence from structure
                        sequence = str(to_sequence(structure)[0][0])
                        seq_features = load_sequence(sequence, device=device)

                        # Predict dynamics with both sequence and structure
                        with torch.no_grad():
                            dynamics_pred = model({
                                "seq_feats": seq_features,
                                "struct_feats": struct_features,
                            })
                            # rmsf = dynamics_pred["rmsf"]
                            # gcc_lmi = dynamics_pred["gcc_lmi"]
                            shp = dynamics_pred["shp"]

                            # data["rmsf"] = rmsf
                            # data["gcc_lmi"] = gcc_lmi
                            data["shp"] = shp

                        entries.append(data)

                except Exception as e:
                    entries.append(data)
                    print(f"Failed to retrieve data from uniprot {uniprot_id}: {e}")
                    continue
    return entries


# for task in ["Contact"]:
#     for split in ["test", "valid", "train"]:
#         lmdb_path = f"LMDB/{task}/foldseek/{split}"
#         print(f"\n🔄 Processing {task} {split} set from {lmdb_path}...")
#
#         dataset_entries = create_dataset_from_lmdb(lmdb_path)
#         print(f"✅ Created new dataset with {len(dataset_entries)} proteins for {split} split")
#
#         save_path = f"LMDB/{task}/dynamic/{split}"
#         os.makedirs(save_path, exist_ok=True)
#         save_lmdb(dataset_entries, save_path)
#         print(f"💾 Saved enhanced dataset to {save_path}")


def filter_multiconf_entries(lmdb_path, multiconf_uniprot_ids=None):
    filtered_entries = []
    total = 0
    errors = 0
    missing_uniprot = 0

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        for key, value in txn.cursor():
            if key == b"length" or key == b"info":
                continue
            total += 1
            try:
                data = json.loads(value.decode('utf-8'))
                if isinstance(data, int):
                    print(f"⚠️ Skipped int entry: key = {key}")
                    errors += 1
                    continue

                if "name_1" in data:
                    uniprot_id_1 = data.get("name_1", None)
                    uniprot_id_2 = data.get("name_2", None)
                    uniprot_id_1 = uniprot_id_1 if "-" not in uniprot_id_1 else uniprot_id_1.split("-")[1]
                    uniprot_id_2 = uniprot_id_2 if "-" not in uniprot_id_2 else uniprot_id_2.split("-")[1]

                    if uniprot_id_1 is None or uniprot_id_2 is None:
                        print(f"⚠️ Missing UniProt ID in key = {key}")
                        missing_uniprot += 1
                        continue

                    if multiconf_uniprot_ids is None or uniprot_id_1 in multiconf_uniprot_ids or uniprot_id_2 in multiconf_uniprot_ids:
                        filtered_entries.append(data)

                else:
                    uniprot_id = data.get("name", None)
                    uniprot_id = uniprot_id if "-" not in uniprot_id else uniprot_id.split("-")[1]

                    if uniprot_id is None:
                        print(f"⚠️ Missing UniProt ID in key = {key}")
                        missing_uniprot += 1
                        continue

                    if multiconf_uniprot_ids is None or uniprot_id in multiconf_uniprot_ids:
                        # print(f"protein {uid} is a multi-conformation protein")
                        filtered_entries.append(data)

            except Exception as e:
                print(f"❌ Error reading key = {key}: {e}")
                errors += 1

    print(f"✅ Done reading: total={total}, kept={len(filtered_entries)}, errors={errors}, missing_uniprot={missing_uniprot}")
    return filtered_entries


def create_humanppi_dataset_from_lmdb(path):
    device = torch.device("cuda:0")
    model = RocketSHP.load_from_checkpoint("v1", strict=False).to(device)

    entries = []
    with lmdb.open(path, readonly=True, lock=False) as env:
        with env.begin() as txn:
            for key, value in txn.cursor():
                if key == b"length" or key == b"info":
                    continue
                try:
                    data = json.loads(value.decode("utf-8"))
                    if isinstance(data, int) or isinstance(value, int):
                        continue

                    uniprot_id_1 = data.get("name_1")
                    uniprot_id_2 = data.get("name_2")

                    uniprot_id_1 = uniprot_id_1 if "-" not in uniprot_id_1 else uniprot_id_1.split("-")[1]
                    uniprot_id_2 = uniprot_id_2 if "-" not in uniprot_id_2 else uniprot_id_2.split("-")[1]

                    for uniprot_id in [uniprot_id_1, uniprot_id_2]:
                        if uniprot_id:
                            save_path = f"example/{uniprot_id}.pdb"
                            if not os.path.exists(save_path):
                                download_alphafold_pdb_v4(uniprot_id, save_path)

                            # Load structure file (PDB)
                            structure = pdb.PDBFile.read(save_path).get_structure()
                            struct_features = load_structure(structure, device=device)

                            # Get sequence from structure
                            sequence = str(to_sequence(structure)[0][0])
                            seq_features = load_sequence(sequence, device=device)

                            # Predict dynamics with both sequence and structure
                            with torch.no_grad():
                                dynamics_pred = model({
                                    "seq_feats": seq_features,
                                    "struct_feats": struct_features,
                                })
                                # rmsf = dynamics_pred["rmsf"]
                                # gcc_lmi = dynamics_pred["gcc_lmi"]
                                shp = dynamics_pred["shp"]

                                # data["rmsf"] = rmsf
                                # data["gcc_lmi"] = gcc_lmi
                                if uniprot_id == uniprot_id_1:
                                    data["shp_1"] = shp
                                else:
                                    data["shp_2"] = shp

                            entries.append(data)

                except Exception as e:
                    entries.append(data)
                    print(f"Failed to retrieve data from uniprot {uniprot_id}: {e}")
                    continue
    return entries


# for task in ["HumanPPI"]:
#     for split in ["train"]:
#         lmdb_path = f"LMDB/{task}/foldseek/{split}"
#         print(f"\n🔄 Processing {task} {split} set from {lmdb_path}...")
#
#         dataset_entries = create_humanppi_dataset_from_lmdb(lmdb_path)
#         print(f"✅ Created new dataset with {len(dataset_entries)} proteins for {split} split")
#
#         save_path = f"LMDB/{task}/dynamic/{split}"
#         os.makedirs(save_path, exist_ok=True)
#         save_lmdb(dataset_entries, save_path)
#         print(f"💾 Saved enhanced dataset to {save_path}")


# Load CoDNaS-Q and extract multiconformation UniProt IDs
# df = pd.read_csv("clean_clusters_codnasQ.csv", sep=';')
#
# # Keep relevant columns only
# df = df[['Query_UniProt_ID', 'Target_UniProt_ID', 'RMSD', 'num_of_conformers']]
#
# # Threshold for defining meaningful structural diversity
# min_thr = 0.4    # Å
# max_thr = 100   # Å
# min_num_of_conformers = 3   # 2 for DeepLoc
# mask = (df['RMSD'] >= min_thr) & (df['RMSD'] <= max_thr) & (df['num_of_conformers'] >= min_num_of_conformers)
# subset = df[mask]
#
# multiconf_ids = set(subset['Query_UniProt_ID']).union(subset['Target_UniProt_ID'])
# print(f"🧬 {len(multiconf_ids):,} UniProt entries have multiple conformers with RMSD in range.")
#
# summary = {}
# for task in ["EC/AF2"]:
#     summary[task] = {}
#     for split in ["test"]:
#         lmdb_path = f"LMDB/{task}/dynamic/{split}"
#         print(f"\n🔄 Processing {task} {split} set from {lmdb_path}...")
#
#         entries = filter_multiconf_entries(lmdb_path, multiconf_ids)
#         count = len(entries)
#         print(f"✅ Found {count} multiconformer proteins")
#
#         summary[task][split] = count
#
#         os.makedirs(f"LMDB/{task}/dynamic/{split}", exist_ok=True)
#         save_lmdb(entries, f"LMDB/{task}/dynamic_test")
#         print(f"💾 Saved Multi-conformation only dataset!")


def npz_to_pdb(npz_path: Path, out_pdb: Path, chain_id="A"):
    data = np.load(npz_path, allow_pickle=True)

    def _write_ca(coords_ca, aatype=None, res_idx=None):
        L = coords_ca.shape[0]
        if res_idx is None:
            res_idx = np.arange(1, L+1, dtype=int)
        if aatype is None:
            # default to GLY for all if not provided; just for PDB header
            res3 = ["GLY"] * L
        else:
            if aatype.dtype.kind in "SU":
                # string codes; normalize to 3-letter
                res3 = []
                for a in aatype:
                    if len(a) == 1:
                        res3.append(struc.AMINO_ACIDS_1TO3.get(a, "UNK"))
                    else:
                        res3.append(a if len(a)==3 else "UNK")
            else:
                # integer indices → 1-letter → 3-letter
                res3 = [struc.AMINO_ACIDS_1TO3[struc.aaindex_to_aa(int(a))] for a in aatype]

        arr = struc.AtomArray(L)
        arr.coord = coords_ca
        arr.atom_name = np.array(["CA"]*L)
        arr.res_name = np.array(res3)
        arr.chain_id = np.array([chain_id]*L)
        arr.res_id = res_idx.astype(int)
        arr.element = np.array(["C"]*L)
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        pdb.PDBFile.write(out_pdb, arr)

    # 1) AlphaFold-style: atom_positions [L,37,3]
    if "atom_positions" in data:
        pos37 = data["atom_positions"]
        aatype = data.get("aatype")
        res_idx = data.get("residue_index")
        _write_ca(pos37[:, 1, :], aatype=aatype, res_idx=res_idx)
        return True

    # 2) Alternative name: atom37_positions
    if "atom37_positions" in data:
        pos37 = data["atom37_positions"]
        aatype = data.get("aatype")
        res_idx = data.get("residue_index")
        _write_ca(pos37[:, 1, :], aatype=aatype, res_idx=res_idx)
        return True

    # 3) Backbone arrays (N/CA/C)
    if "CA" in data:
        coords_ca = data["CA"]
        aatype = data.get("aatype")
        res_idx = data.get("residue_index")
        _write_ca(coords_ca, aatype=aatype, res_idx=res_idx)
        return True

    # 4) Last resort: packed atoms with names
    if "atom_names" in data and "atom_coords" in data:
        # Expect shapes [L,A] and [L,A,3]
        names = data["atom_names"]
        coords = data["atom_coords"]
        # find CA index per residue
        ca_coords = []
        for r in range(coords.shape[0]):
            row_names = names[r]
            idxs = np.where(row_names == "CA")[0]
            if len(idxs) == 0:
                # cannot salvage this residue
                return False
            ca_coords.append(coords[r, idxs[0], :])
        coords_ca = np.stack(ca_coords, axis=0)
        aatype = data.get("aatype")
        res_idx = data.get("residue_index")
        _write_ca(coords_ca, aatype=aatype, res_idx=res_idx)
        return True

    return False


# Example: iterate all NPZ shards for a protein
def convert_shard_folder_to_pdbs(npz_dir: Path, out_dir: Path):
    for npz_path in sorted(npz_dir.glob("batch_*.npz")):
        out_pdb = out_dir / (npz_path.stem + ".pdb")
        npz_to_pdb(npz_path, out_pdb)


def foldseek_tokens_for_pdb(foldseek_bin: str = "bin/foldseek", pdb_path: str = "", chain: str = "A", plddt_mask = False):
    parsed = get_struc_seq(foldseek_bin, pdb_path, [chain], plddt_mask=plddt_mask)
    if chain not in parsed:
        raise KeyError(f"Chain {chain} not found in {pdb_path}")

    row = parsed[chain]
    # row may be (seq, foldseek_seq) or (seq, foldseek_seq, combined_seq) or longer
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        raise ValueError(f"Unexpected get_struc_seq output for {pdb_path}, chain {chain}: {type(row)} len={len(row) if hasattr(row, '__len__') else 'n/a'}")

    seq, foldseek_seq = row[0], row[1]  # take the first two
    return seq, foldseek_seq


def tokens_to_distribution(token_strs, eps=1e-3):
    """
    token_strs: list of K strings, each length L (3Di tokens, letters A..T)
    returns: probs [L, 20] in the fixed alphabet order A..T
    """
    assert len(token_strs) > 0
    L = len(token_strs[0])
    assert all(len(s) == L for s in token_strs), "Inconsistent token lengths"

    counts = np.zeros((L, 20), dtype=np.int32)
    for s in token_strs:
        for i, c in enumerate(s):
            j = THREEDI_INDEX.get(c)
            if j is not None:      # ignore unexpected chars/gaps
                counts[i, j] += 1

    probs = (counts + eps) / (counts.sum(axis=1, keepdims=True) + eps * 20)
    return probs.astype(np.float32), np.array(THREEDI_ALPHABET)


def build_shp_like_from_npz_shards(npz_dir, out_dir, foldseek_bin="bin/foldseek", chain="A"):
    npz_dir, out_dir = Path(npz_dir), Path(out_dir)
    pdb_dir = out_dir / "pdb_frames"
    shp_dir = out_dir / "shp_like"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    shp_dir.mkdir(parents=True, exist_ok=True)

    # 1) NPZ -> PDBs
    convert_shard_folder_to_pdbs(npz_dir, pdb_dir)

    # 2) PDB -> 3Di tokens
    token_strs = []
    pdb_dir = Path(pdb_dir)
    for pdb_file in sorted(pdb_dir.glob("*.pdb")):
        _, threeDi = foldseek_tokens_for_pdb(foldseek_bin, str(pdb_file), chain=chain, plddt_mask=False)
        token_strs.append(threeDi)

    # 3) Tokens -> LxA distribution
    probs, alphabet = tokens_to_distribution(token_strs)
    np.savez(shp_dir / "shp_like.npz", probs=probs, alphabet=np.array(alphabet))
    return probs, alphabet


def create_dataset_with_bioemu_from_lmdb(path):
    device = torch.device("cuda:0")
    model = RocketSHP.load_from_checkpoint("v1", strict=False).to(device)

    entries = []
    with lmdb.open(path, readonly=True, lock=False) as env:
        with env.begin() as txn:
            for key, value in txn.cursor():
                if key == b"length":
                    continue
                try:
                    data = json.loads(value.decode("utf-8"))
                    if isinstance(data, int) or isinstance(value, int):
                        continue

                    uniprot_id = data.get("name")
                    uniprot_id = uniprot_id if "-" not in uniprot_id else uniprot_id.split("-")[1]

                    if uniprot_id:
                        save_path = f"example/{uniprot_id}.pdb"
                        if not os.path.exists(save_path):
                            download_alphafold_pdb_v4(uniprot_id, save_path)

                        # Load structure file (PDB)
                        structure = pdb.PDBFile.read(save_path).get_structure()
                        struct_features = load_structure(structure, device=device)

                        # Get sequence from structure
                        sequence = str(to_sequence(structure)[0][0])
                        seq_features = load_sequence(sequence, device=device)

                        path_bioemu = f'/home/kalifadan/DynamicsPLM/bioemu/{uniprot_id}'
                        if not os.path.exists(path_bioemu):
                            sample(sequence=sequence, num_samples=10, output_dir=path_bioemu)

                        path_bioemu = Path(path_bioemu)
                        pdb_frames = sorted(path_bioemu.glob("*.pdb"))

                        # If none, convert NPZ shards to PDBs
                        if len(pdb_frames) == 0:
                            npz_list = sorted(path_bioemu.glob("batch_*.npz"))
                            if len(npz_list) == 0:
                                data.setdefault("errors", []).append("No BioEmu outputs found.")
                                entries.append(data);
                                continue
                            out_pdb_dir = path_bioemu / "pdb_frames"
                            out_pdb_dir.mkdir(exist_ok=True, parents=True)
                            for npz_path in npz_list:
                                out_pdb = out_pdb_dir / f"{npz_path.stem}.pdb"
                                if not out_pdb.exists():
                                    ok = npz_to_pdb(npz_path, out_pdb)
                                    if not ok:
                                        # if we cannot find CA or schema unsupported, skip this shard
                                        continue
                            pdb_frames = sorted(out_pdb_dir.glob("*.pdb"))
                            if len(pdb_frames) == 0:
                                data.setdefault("errors", []).append(
                                    "NPZ→PDB conversion failed (no CA/backbone found).")
                                entries.append(data)
                                continue

                        # Foldseek tokenization for each frame
                        token_strs = []
                        for f in pdb_frames:
                            try:
                                # For generated frames there is no pLDDT → plddt_mask=False
                                _, three_di = foldseek_tokens_for_pdb("bin/foldseek", str(f), chain="A",
                                                                       plddt_mask=False)
                                token_strs.append(three_di)
                            except Exception as e:
                                # skip bad frame but continue
                                continue

                        if len(token_strs) == 0:
                            data.setdefault("errors", []).append("Foldseek tokenization produced no sequences.")
                            entries.append(data)
                            continue

                        # Aggregate to LxK distribution
                        probs, alphabet = tokens_to_distribution(token_strs)
                        # (Optional) enforce same L as input sequence by trimming/padding if needed
                        L = len(sequence)
                        if probs.shape[0] != L:
                            # simple trim/pad; you may want a smarter alignment if mismatched
                            if probs.shape[0] > L:
                                probs = probs[:L]
                            else:
                                pad = np.tile(probs[-1:], (L - probs.shape[0], 1))
                                probs = np.vstack([probs, pad])

                        data["shp"] = probs.astype(np.float16).tolist()
                        data["K"] = int(probs.shape[1])
                        data["L"] = int(L)
                        data["shp_source"] = "bioemu"

                        entries.append(data)

                except Exception as e:
                    entries.append(data)
                    print(f"Failed to retrieve data from uniprot {uniprot_id}: {e}")
                    continue
    return entries


for task in ["MetalIonBinding/AF2"]:        # for example: "DeepLoc/cls10", "EC/AF2"]:
    for split in ["test", "valid", "train"]:
        lmdb_path = f"LMDB/{task}/foldseek/{split}"
        print(f"\n🔄 Processing {task} {split} set from {lmdb_path}...")

        dataset_entries = create_dataset_with_bioemu_from_lmdb(lmdb_path)
        print(f"✅ Created new dataset with {len(dataset_entries)} proteins for {split} split")

        save_path = f"LMDB/{task}/dynamic_bioemu/{split}"
        os.makedirs(save_path, exist_ok=True)
        save_lmdb(dataset_entries, save_path)
        print(f"💾 Saved enhanced dataset to {save_path}")
