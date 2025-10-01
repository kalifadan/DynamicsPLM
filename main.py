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
import time

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


_HAS_BIOTITE = True

_AA1TO3_FALLBACK = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE",
    "G":"GLY","H":"HIS","I":"ILE","K":"LYS","L":"LEU",
    "M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG",
    "S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR",
    "X":"GLY","U":"SEC","O":"PYL"
}


def _aa1_to_aa3_array(aa_1letter_iterable):
    if _HAS_BIOTITE and hasattr(struc, "AMINO_ACIDS_1TO3"):
        m = struc.AMINO_ACIDS_1TO3
        return np.array([m.get(a.upper(), "GLY") for a in aa_1letter_iterable], dtype="U3")
    return np.array([_AA1TO3_FALLBACK.get(a.upper(), "GLY") for a in aa_1letter_iterable], dtype="U3")


def _coerce_sequence_to_1letter(seq_obj):
    # str
    if isinstance(seq_obj, str):
        return seq_obj
    # bytes
    if isinstance(seq_obj, (bytes, bytearray)):
        return bytes(seq_obj).decode("utf-8")
    # list/tuple
    if isinstance(seq_obj, (list, tuple)):
        if len(seq_obj) == 1 and isinstance(seq_obj[0], str) and len(seq_obj[0]) > 1:
            return seq_obj[0]
        return "".join(str(x) for x in seq_obj)

    arr = np.asarray(seq_obj)
    # integer indices
    if arr.dtype.kind in ("i", "u"):
        if _HAS_BIOTITE and hasattr(struc, "aaindex_to_aa"):
            return "".join(struc.aaindex_to_aa(int(i)) for i in arr.reshape(-1))
        raise TypeError("Sequence is integer indices but no aaindex_to_aa() available.")
    # strings/bytes/object arrays
    if arr.dtype.kind in ("U", "S", "O"):
        arr = arr.squeeze()
        if arr.shape == ():
            return str(arr.item())
        if arr.shape == (1,):
            return str(arr[0])
        tokens = [str(x) for x in arr.tolist()]
        # join tokens (works for 1-letter tokens or a single full string split up)
        return "".join(tokens)
    raise TypeError(f"Unsupported sequence dtype/shape: dtype={arr.dtype}, shape={arr.shape}")


def _format_atom_line(serial, res3, chain_id, resseq, x, y, z, occ=1.00, b=0.00, element="C"):
    # PDB ATOM line (columns per PDB spec, CA only)
    # 1-6  "ATOM  "
    # 7-11 serial, 13-16 atom name, 18 altLoc, 18-20 resName, 22 chainID,
    # 23-26 resSeq, 31-38 x, 39-46 y, 47-54 z, 55-60 occ, 61-66 temp
    return (
        f"ATOM  {serial:5d}  CA  {res3:>3s} {chain_id}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          {element:>2s}\n"
    )


def _write_pdb(out_pdb: Path, coords_models, res3, chain_id="A", start_res_id=1):
    """
    coords_models : list of np.ndarray, each (L,3) for one model
    res3          : array of len L with 3-letter codes
    """
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb, "w", encoding="utf-8", newline="\n") as f:
        atom_serial = 1
        for m_idx, coords in enumerate(coords_models):
            if len(coords_models) > 1:
                f.write(f"MODEL     {m_idx+1:4d}\n")
            L = coords.shape[0]
            # sanitize coords
            coords = np.asarray(coords, dtype=float)
            coords = np.where(np.isfinite(coords), coords, 0.0)

            for i in range(L):
                x, y, z = coords[i]
                resseq = start_res_id + i
                f.write(_format_atom_line(atom_serial, res3[i], chain_id, resseq, x, y, z))
                atom_serial += 1

            if len(coords_models) > 1:
                f.write("ENDMDL\n")
        f.write("END\n")


def npz_to_pdb(npz_path: Path, out_pdb: Path, chain_id="A"):
    """
    Convert BioEmu NPZ (expects keys: 'pos', 'sequence') to CA-only PDB.
    - pos: (L,3) or (K,L,3)
    - sequence: str / 1-element array[str] / list[str] / int indices (if Biotite available)
    """
    data = np.load(npz_path, allow_pickle=True)
    if "pos" not in data or "sequence" not in data:
        raise KeyError(f"{npz_path} must contain 'pos' and 'sequence'")

    pos = np.asarray(data["pos"])
    seq_1 = _coerce_sequence_to_1letter(data["sequence"])
    L = len(seq_1)

    # Validate shapes and collect models
    if pos.ndim == 2:
        if pos.shape != (L, 3):
            raise ValueError(f"'pos' shape {pos.shape} does not match sequence length {L}")
        coords_models = [pos]
    elif pos.ndim == 3:
        K, Lp, D = pos.shape
        if (Lp, D) != (L, 3):
            raise ValueError(f"'pos' shape {pos.shape} does not match (L,3) with L={L}")
        coords_models = [pos[k] for k in range(K)]
    else:
        raise ValueError(f"Unsupported 'pos' ndim={pos.ndim}; expected (L,3) or (K,L,3)")

    # Map sequence to 3-letter names
    res3 = _aa1_to_aa3_array(list(seq_1))

    # Write plain-text PDB
    _write_pdb(out_pdb, coords_models, res3, chain_id=chain_id)
    return True


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
                        sequence = str(to_sequence(structure)[0][0])

                        path_bioemu = f'/home/kalifadan/DynamicsPLM/bioemu/{uniprot_id}'
                        if os.path.exists(path_bioemu) and len(sorted(Path(path_bioemu).glob("*.npz"))) > 0:
                            print(f"skip sample structures for protein {uniprot_id}")
                        else:
                            sample(sequence=sequence, num_samples=10, output_dir=path_bioemu)
                            time.sleep(0.1)

                        path_bioemu = Path(path_bioemu)
                        npz_list = sorted(path_bioemu.glob("batch_*.npz"))
                        if len(npz_list) == 0:
                            print("No BioEmu outputs found.")
                            entries.append(data)
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
                            print("NPZ→PDB conversion failed (no CA/backbone found).")
                            entries.append(data)
                            continue
                        else:
                            print(f"Found {len(pdb_frames)} valid pdb frames for {uniprot_id}")

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
                            print("Foldseek tokenization produced no sequences.")
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
                        print(f"Success producing shp for {uniprot_id}")

                        data["shp_source"] = "bioemu"
                        entries.append(data)

                except Exception as e:
                    entries.append(data)
                    print(f"Failed to retrieve data from uniprot {uniprot_id}: {e}")
                    continue
    return entries


for task in ["MetalIonBinding/AF2", "DeepLoc/cls10", "EC/AF2"]:
    for split in ["test", "valid", "train"]:
        lmdb_path = f"LMDB/{task}/foldseek/{split}"
        print(f"\n🔄 Processing {task} {split} set from {lmdb_path}...")

        dataset_entries = create_dataset_with_bioemu_from_lmdb(lmdb_path)
        print(f"✅ Created new dataset with {len(dataset_entries)} proteins for {split} split")

        save_path = f"LMDB/{task}/dynamic_bioemu/{split}"
        os.makedirs(save_path, exist_ok=True)
        save_lmdb(dataset_entries, save_path)
        print(f"💾 Saved enhanced dataset to {save_path}")
