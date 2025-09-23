from __future__ import annotations
import requests
import os
import lmdb
import json
import torch
import os
import numpy as np
# from rocketshp import RocketSHP, load_sequence, load_structure
# from biotite.structure.io import pdb
# from biotite.structure import to_sequence
from pathlib import Path
import math
import pandas as pd


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


def delete_keys_from_lmdb(data_lmdb_path, keys_to_delete):
    env = lmdb.open(data_lmdb_path, map_size=1 << 40, max_dbs=1)
    with env.begin(write=True) as txn:
        length = txn.get(b"length")
        if length is None:
            print(f"[⚠️] No 'length' key found in {data_lmdb_path}")
        else:
            print(f"[📦] {data_lmdb_path} contains {int(length)} entries")
        for key in keys_to_delete:
            key_b = str(key).encode()
            deleted = txn.delete(key_b)
            if deleted:
                print(f"Deleted key: {key}")
            else:
                print(f"Key not found: {key}")


def validate_lmdb_length(lmdb_path):
    with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
        with env.begin() as txn:
            # Get stored length value
            length_raw = txn.get(b"length")
            if length_raw is None:
                print(f"No 'length' key in {lmdb_path}")
                return
            stored_length = int(length_raw.decode())

            # Count all real data entries (excluding 'length')
            cursor = txn.cursor()
            actual_count = sum(1 for k, _ in cursor if k != b"length")

            print(f"[📂] LMDB: {lmdb_path}")
            print(f"  Stored length: {stored_length}")
            print(f"  Actual entries: {actual_count}")

            if stored_length != actual_count:
                print(f"MISMATCH: difference = {stored_length - actual_count}")
            else:
                print(f"LENGTH MATCHES")


def decrement_lmdb_length(lmdb_path, amount=2):
    with lmdb.open(lmdb_path, map_size=1 << 40) as env:
        with env.begin(write=True) as txn:
            length_raw = txn.get(b"length")
            if length_raw is None:
                print(f"'length' key not found in {lmdb_path}")
                return

            old_length = int(length_raw.decode())
            new_length = max(0, old_length - amount)  # Prevent negative length

            txn.put(b"length", str(new_length).encode())
            print(f"Updated 'length': {old_length} → {new_length}")


def fill_missing_lmdb_entries(original_path, dynamic_path):
    print(f"🔍 Checking for missing entries in {dynamic_path}")
    device = torch.device("cuda:0")
    model = RocketSHP.load_from_checkpoint("v1", strict=False).to(device)

    with lmdb.open(original_path, readonly=True, lock=False) as orig_env, \
         lmdb.open(dynamic_path, map_size=1 << 40) as dyn_env:

        with orig_env.begin() as orig_txn, dyn_env.begin(write=True) as dyn_txn:
            # Get original length
            raw_len = dyn_txn.get(b"length")
            if raw_len is None:
                raise ValueError(f"No 'length' key in {dynamic_path}")
            length = int(raw_len) + 1

            recovered = 0

            for i in range(length):
                key = str(i).encode()

                if dyn_txn.get(key) is None:
                    # missing key → copy from original
                    print("missing key:", key)
                    value = orig_txn.get(key)
                    if value is not None:

                        data = json.loads(value.decode("utf-8"))
                        uniprot_id = data.get("name")

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
                                rmsf = dynamics_pred["rmsf"]
                                gcc_lmi = dynamics_pred["gcc_lmi"]
                                shp = dynamics_pred["shp"]

                                data["rmsf"] = rmsf
                                data["gcc_lmi"] = gcc_lmi
                                data["shp"] = shp

                        def convert(obj):
                            if isinstance(obj, torch.Tensor):
                                return obj.tolist()
                            elif isinstance(obj, dict):
                                return {k: convert(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [convert(x) for x in obj]
                            else:
                                return obj

                        clean_entry = convert(data)
                        dyn_txn.put(key, json.dumps(clean_entry).encode())

                        recovered += 1
                        print(f"[+{i}] Recovered missing entry {i}")
                    else:
                        print(f"[x] Original also missing entry {i}")

            print(f"✅ Filled {recovered} missing entries into {dynamic_path}")


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

import re

# Load CoDNaS-Q and extract multiconformation UniProt IDs
df = pd.read_csv("clusters_codnasQ.csv", sep=';')

# --- Find and normalize the 'num_of_conformers' column name (handles messy header) ---
candidates = [c for c in df.columns if "num_of_conformer" in c.lower()]
if not candidates:
    raise KeyError("Couldn't find a 'num_of_conformers' column (even a messy one).")
bad_col = candidates[0]
df.rename(columns={bad_col: "num_of_conformers"}, inplace=True)


# --- Clean 'num_of_conformers' values like '7,,,,,,,', '  12  ' -> 7 / 12 ---
def parse_conf(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip()
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else pd.NA


df["num_of_conformers"] = df["num_of_conformers"].map(parse_conf).astype("Int64")

# Keep relevant columns only
df = df[['Query_UniProt_ID', 'Target_UniProt_ID', 'RMSD', 'num_of_conformers']]

# Threshold for defining meaningful structural diversity
min_thr = 0.4      # 0.8     # 0.4    # Å
max_thr = 100   # 1.5   # Å
min_num_of_conformers = 3   # 3       # 2
mask = (df['RMSD'] >= min_thr) & (df['RMSD'] <= max_thr) & (df['num_of_conformers'] >= min_num_of_conformers)
subset = df[mask]

subset.to_csv("clean_clusters_codnasQ.csv", index=False)

# Set of UniProt IDs with >1 conformation AND RMSD > threshold
multiconf_ids = set(subset['Query_UniProt_ID']).union(subset['Target_UniProt_ID'])
print(f"🧬 {len(multiconf_ids):,} UniProt entries have multiple conformers with RMSD in range.")


summary = {}
# for task in ["HumanPPI", "EC/AF2", "MetalIonBinding/AF2", "DeepLoc/cls2", "DeepLoc/cls10"]:
for task in ["EC/AF2"]:
    summary[task] = {}
    for split in ["test"]:      # , "valid", "train"]:
        lmdb_path = f"LMDB/{task}/dynamic/{split}"
        print(f"\n🔄 Processing {task} {split} set from {lmdb_path}...")

        entries = filter_multiconf_entries(lmdb_path, multiconf_ids)
        count = len(entries)
        print(f"✅ Found {count} multiconformer proteins")

        summary[task][split] = count

        os.makedirs(f"LMDB/{task}/dynamic/{split}", exist_ok=True)
        save_lmdb(entries, f"LMDB/{task}/dynamic_test")
        print(f"💾 Saved Multi-conformation only dataset!")

# Print final summary table
print("\n📊 Summary of multiconformer proteins per task and split:")
print("{:<25} {:>8} {:>8} {:>8} {:>8}".format("Task", "Test", "Valid", "Train", "Total"))
print("-" * 60)
for task, splits in summary.items():
    test = splits.get("test", 0)
    # valid = splits.get("valid", 0)
    # train = splits.get("train", 0)
    # total = test + valid + train
    print("{:<25} {:>8}".format(task, test))
#


# import pandas as pd
# import re, csv
#
# # ------------------------------
# # Configurable thresholds (tweak if needed)
# # ------------------------------
# MIN_CONFS = 5                 # bona-fide dynamics: multiple experimental conformers
# RMSD_MIN, RMSD_MAX = 1.5, 5.0 # non-trivial motion, avoid misalignments
# MAX_RES   = 3.0               # at least one high-quality structure
# MIN_COVER = 0.80              # aligned-domain coverage
# MAX_TDE   = 2.5               # reliable superposition quality
# REQUIRE_LIGAND_OR_ASSEMBLY_CHANGE = True
# REQUIRE_METAL_CONTEXT = True  # helpful for function-coupled dynamics
#
# # ------------------------------
# # Load CoDNaS-Q (robust to malformed rows/quotes)
# # ------------------------------
# df = pd.read_csv("clusters_codnasQ.csv",
#                  sep=";", engine="python",
#                  quoting=csv.QUOTE_NONE,
#                  on_bad_lines="skip")
#
# # --- Find and normalize the 'num_of_conformers' column name (handles messy header) ---
# candidates = [c for c in df.columns if "num_of_conformer" in c.lower()]
# if not candidates:
#     raise KeyError("Couldn't find a 'num_of_conformers' column (even a messy one).")
# bad_col = candidates[0]
# df.rename(columns={bad_col: "num_of_conformers"}, inplace=True)
#
# # --- Clean 'num_of_conformers' values like '7,,,,,,,', '  12  ' -> 7 / 12 ---
# def parse_conf(v):
#     if pd.isna(v):
#         return pd.NA
#     m = re.search(r"\d+", str(v).strip())
#     return int(m.group(0)) if m else pd.NA
#
# df["num_of_conformers"] = df["num_of_conformers"].map(parse_conf).astype("Int64")
#
# # --- Coerce numeric columns (silently to NaN if malformed) ---
# for col in [
#     "RMSD",
#     "Query_resolution", "Target_resolution",
#     "Query_cover_based_on_alignment_length", "Target_cover_based_on_alignment_length",
#     "Typical_distance_error"
# ]:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# # --- Prepare ligand / assembly state-change signals ---
# for col in [
#     "Query_ligands","Target_ligands",
#     "Biological_Assembly_query","Biological_Assembly_target"
# ]:
#     if col in df.columns:
#         df[col] = df[col].astype(str)
#
# def ligand_changed(q, t):
#     q = (q or "").strip()
#     t = (t or "").strip()
#     if q == "" and t == "":
#         return False
#     return q != t
#
# def assembly_changed(q, t):
#     return (q or "") != (t or "")
#
# METALS = {"ZN","MG","MN","FE","CU","CO","CA","NI","K","NA","CD"}
# def has_metal(s):
#     toks = re.split(r'[;, ]+', (s or ""))
#     toks = [t.strip().upper() for t in toks if t.strip()]
#     return any(t in METALS for t in toks)
#
# df["ligand_change"]   = df.apply(lambda r: ligand_changed(r.get("Query_ligands"), r.get("Target_ligands")), axis=1)
# df["assembly_change"] = df.apply(lambda r: assembly_changed(r.get("Biological_Assembly_query"), r.get("Biological_Assembly_target")), axis=1)
# df["has_metal_any"]   = df["Query_ligands"].apply(has_metal) | df["Target_ligands"].apply(has_metal)
#
# # ------------------------------
# # Build a per-side long table, then aggregate per UniProt
# # ------------------------------
# q = df[[
#     "Query_UniProt_ID", "RMSD", "num_of_conformers",
#     "Query_resolution", "Query_cover_based_on_alignment_length",
#     "Typical_distance_error", "ligand_change", "assembly_change", "has_metal_any"
# ]].rename(columns={
#     "Query_UniProt_ID": "UniProt_ID",
#     "Query_resolution": "resolution",
#     "Query_cover_based_on_alignment_length": "cover_on_align",
# })
#
# t = df[[
#     "Target_UniProt_ID", "RMSD", "num_of_conformers",
#     "Target_resolution", "Target_cover_based_on_alignment_length",
#     "Typical_distance_error", "ligand_change", "assembly_change", "has_metal_any"
# ]].rename(columns={
#     "Target_UniProt_ID": "UniProt_ID",
#     "Target_resolution": "resolution",
#     "Target_cover_based_on_alignment_length": "cover_on_align",
# })
#
# long = pd.concat([q, t], ignore_index=True)
# long = long.dropna(subset=["UniProt_ID"])
# long["UniProt_ID"] = long["UniProt_ID"].astype(str)
#
# agg = (long.groupby("UniProt_ID", as_index=False)
#            .agg(max_conformers=("num_of_conformers", "max"),
#                 max_RMSD=("RMSD", "max"),
#                 min_resolution=("resolution", "min"),
#                 max_cover_on_align=("cover_on_align", "max"),
#                 min_TDE=("Typical_distance_error", "min"),
#                 any_ligand_change=("ligand_change", "max"),
#                 any_assembly_change=("assembly_change", "max"),
#                 any_has_metal=("has_metal_any", "max")))
#
# # ------------------------------
# # Apply defensible "dynamic" filter
# # ------------------------------
# mask = (
#     (agg["max_conformers"] >= MIN_CONFS) &
#     (agg["max_RMSD"] >= RMSD_MIN) & (agg["max_RMSD"] <= RMSD_MAX) &
#     (agg["min_resolution"] <= MAX_RES) &
#     (agg["max_cover_on_align"] >= MIN_COVER) &
#     (agg["min_TDE"] <= MAX_TDE)
# )
#
# if REQUIRE_LIGAND_OR_ASSEMBLY_CHANGE:
#     mask &= (agg["any_ligand_change"] | agg["any_assembly_change"])
#
# if REQUIRE_METAL_CONTEXT:
#     mask &= (agg["any_has_metal"] == True)
#
# dynamic_subset = agg[mask].copy()
#
# # Save per-protein dynamic features (nice for Methods/SI)
# dynamic_subset.to_csv("clean_clusters_codnasQ_dynamic_subset.csv", index=False)
#
# # Final “query” set of UniProt IDs
# multiconf_ids = set(dynamic_subset["UniProt_ID"])
# print(f"🧬 {len(multiconf_ids):,} UniProt entries in dynamic subset "
#       f"(confs≥{MIN_CONFS}, RMSD∈[{RMSD_MIN},{RMSD_MAX}]Å, res≤{MAX_RES}Å, "
#       f"cover≥{MIN_COVER}, TDE≤{MAX_TDE}, "
#       f"{'lig/assembly change, ' if REQUIRE_LIGAND_OR_ASSEMBLY_CHANGE else ''}"
#       f"{'metal context' if REQUIRE_METAL_CONTEXT else 'no metal req.'}).")
#
#
# summary = {}
# # "HumanPPI"
# for task in ["EC/AF2", "MetalIonBinding/AF2", "DeepLoc/cls2", "DeepLoc/cls10"]:
#     summary[task] = {}
#     for split in ["test"]:      # , "valid", "train"]:
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
#
#

# from pathlib import Path
#
#
# def _extract_uid(data):
#     uid = data.get("name")
#     if uid is None:
#         return None
#     return uid.split("-")[1] if "-" in uid else uid
#
#
# def _get_shp(data):
#     shp = data.get("shp")
#     if shp is None:
#         shp = (data.get("dynamic_features") or {}).get("shp")
#     return shp
#
#
# def _coerce_to_LK(shp, k_candidates=(20, 21)):
#     """
#     Coerce SHP array to shape (rows, K).
#     If ndim>=3, move K-axis to last and flatten all other axes.
#     If (K, L) arrives, transpose to (L, K).
#     If (K,) arrives, make it (1, K).
#     Returns None if no K-like axis is found.
#     """
#     A = np.asarray(shp, dtype=np.float64)
#     if A.size == 0:
#         return None
#
#     if A.ndim == 1:
#         return A[None, :] if A.shape[0] in k_candidates else None
#
#     if A.ndim == 2:
#         if A.shape[0] in k_candidates and A.shape[1] not in k_candidates:
#             A = A.T  # (L, K)
#         elif A.shape[1] not in k_candidates and A.shape[0] not in k_candidates:
#             return None
#         return A
#
#     # A.ndim >= 3
#     # find a K axis by size
#     k_axis = None
#     for ax, sz in enumerate(A.shape):
#         if sz in k_candidates:
#             k_axis = ax
#             break
#     if k_axis is None:
#         return None
#
#     A = np.moveaxis(A, k_axis, -1)   # (..., K)
#     K = A.shape[-1]
#     A = A.reshape(-1, K)             # (rows, K)
#     return A
#
#
# def _rowwise_entropy(P, eps=1e-12):
#     P = np.asarray(P, dtype=np.float64)
#     if P.ndim != 2 or P.size == 0:
#         return None
#     P = np.clip(P, eps, None)
#     P = P / P.sum(axis=1, keepdims=True)
#     return -np.sum(P * np.log(P), axis=1)
#
#
# def protein_variety_score(shp_array, normalize=True):
#     """
#     Mean row entropy after coercing to (rows, K). Higher = more variety.
#     """
#     P = _coerce_to_LK(shp_array)
#     if P is None:
#         return None
#     H = _rowwise_entropy(P)
#     if H is None:
#         return None
#     score = float(np.mean(H))
#     if normalize:
#         score /= np.log(P.shape[1])  # 0..1
#     return score
#
#
# def rank_proteins_by_shp_variety(dataset_entries, save_path=None, normalize=True):
#     rows = []
#     n_total = 0
#     n_used = 0
#     n_invalid = 0
#
#     for data in dataset_entries:
#         n_total += 1
#         uid = _extract_uid(data)
#         shp = _get_shp(data)
#         if uid is None or shp is None:
#             n_invalid += 1
#             continue
#
#         P = _coerce_to_LK(shp)
#         if P is None or P.ndim != 2 or P.size == 0:
#             print(f"skip {uid}: SHP shape {np.shape(shp)} cannot be coerced to (rows,K)")
#             n_invalid += 1
#             continue
#
#         score = protein_variety_score(shp, normalize=normalize)
#         if score is None or not np.isfinite(score):
#             print(f"skip {uid}: score invalid for SHP shape {P.shape}")
#             n_invalid += 1
#             continue
#
#         rows.append((uid, score, P.shape[0], P.shape[1]))  # uid, score, rows, K
#         n_used += 1
#
#     rows.sort(key=lambda x: x[1], reverse=True)
#
#     ranked = [
#         {"rank": i, "uniprot_id": uid, "variety_score": score, "n_rows": L, "n_states": K}
#         for i, (uid, score, L, K) in enumerate(rows, start=1)
#     ]
#
#     if save_path:
#         save_path = Path(save_path)
#         if save_path.exists() and save_path.is_dir():
#             save_path = save_path / "shp_variety_ranking.tsv"
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(save_path, "w") as f:
#             f.write("rank\tuniprot_id\tvariety_score\tn_rows\tn_states\n")
#             for r in ranked:
#                 f.write(f"{r['rank']}\t{r['uniprot_id']}\t{r['variety_score']:.6f}\t{r['n_rows']}\t{r['n_states']}\n")
#
#     print(f"✅ Variety ranking: used={n_used}/{n_total}, invalid={n_invalid}")
#     return ranked
#
#
# def load_top10_uniprots(ranking_tsv: str | Path) -> set[str]:
#     """
#     Load the ranking TSV (must include 'uniprot_id' and 'rank') and
#     return the top-20% UniProt IDs as a set.
#     """
#     ranking_tsv = Path(ranking_tsv)
#     if not ranking_tsv.exists():
#         raise FileNotFoundError(f"Ranking file not found: {ranking_tsv}")
#
#     df = pd.read_csv(ranking_tsv, sep="\t")
#     required_cols = {"uniprot_id", "rank"}
#     missing = required_cols - set(df.columns)
#     if missing:
#         raise ValueError(f"{ranking_tsv} missing required columns: {sorted(missing)}")
#
#     # Ensure numeric rank and sort ascending (1 = best)
#     df["rank"] = pd.to_numeric(df["rank"], errors="raise")
#     df = df.sort_values("rank", ascending=True).reset_index(drop=True)
#
#     n_total = len(df)
#     if n_total == 0:
#         return set()
#
#     # Keep at least 1; ceil to avoid rounding down tiny sets.
#     k_top = max(1, math.ceil(0.10 * n_total))
#     top_df = df.iloc[:k_top]
#
#     return set(top_df["uniprot_id"].astype(str))

#
# if __name__ == "__main__":
#     for task in ["EC/AF2", "DeepLoc/cls10", "DeepLoc/cls2", "MetalIonBinding/AF2"]:
#         for split in ["test"]:
#             lmdb_path = f"LMDB/{task}/dynamic/{split}"
#             print(f"\n🔄 Processing {task} {split} set from {lmdb_path}...")
#
#             # Load full dataset (no filtering yet)
#             dataset_entries = filter_multiconf_entries(lmdb_path, None)
#             print(f"✅ Loaded dataset with {len(dataset_entries)} proteins for {split} split")
#
#             # Rank by SHP variety and save TSV
#             out_tsv = f"results/{task}/{split}_shp_variety_ranking.tsv"
#             ranking = rank_proteins_by_shp_variety(dataset_entries, save_path=out_tsv)
#
#             print("Top 5 most dynamic:")
#             for r in ranking[:5]:
#                 print(r)
#
#             top20_ids = load_top10_uniprots(out_tsv)
#             print(f"📈 Selected top 20%: {len(top20_ids)} proteins")
#
#             # Filter entries to top-20% IDs
#             entries = filter_multiconf_entries(lmdb_path, top20_ids)
#             print(f"🔎 Found {len(entries)} entries in top-20% subset")
#
#             # Save to new LMDB (fix: use '{split}_dynamic', not '{split}_dynamics')
#             out_lmdb = f"LMDB/{task}/dynamic/{split}_dynamic"
#             save_lmdb(entries, out_lmdb)
#             print(f"💾 Saved top-20% subset to: {out_lmdb}")
