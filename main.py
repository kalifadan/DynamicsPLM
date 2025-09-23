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

# from rocketshp import RocketSHP, load_sequence, load_structure
# from biotite.structure.io import pdb
# from biotite.structure import to_sequence


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


import re

# Load CoDNaS-Q and extract multiconformation UniProt IDs
df = pd.read_csv("clean_clusters_codnasQ.csv", sep=';')

# Keep relevant columns only
df = df[['Query_UniProt_ID', 'Target_UniProt_ID', 'RMSD', 'num_of_conformers']]

# Threshold for defining meaningful structural diversity
min_thr = 0.4    # Å
max_thr = 100   # Å
min_num_of_conformers = 3   # 2 for DeepLoc
mask = (df['RMSD'] >= min_thr) & (df['RMSD'] <= max_thr) & (df['num_of_conformers'] >= min_num_of_conformers)
subset = df[mask]

multiconf_ids = set(subset['Query_UniProt_ID']).union(subset['Target_UniProt_ID'])
print(f"🧬 {len(multiconf_ids):,} UniProt entries have multiple conformers with RMSD in range.")

summary = {}
for task in ["EC/AF2"]:
    summary[task] = {}
    for split in ["test"]:
        lmdb_path = f"LMDB/{task}/dynamic/{split}"
        print(f"\n🔄 Processing {task} {split} set from {lmdb_path}...")

        entries = filter_multiconf_entries(lmdb_path, multiconf_ids)
        count = len(entries)
        print(f"✅ Found {count} multiconformer proteins")

        summary[task][split] = count

        os.makedirs(f"LMDB/{task}/dynamic/{split}", exist_ok=True)
        save_lmdb(entries, f"LMDB/{task}/dynamic_test")
        print(f"💾 Saved Multi-conformation only dataset!")
