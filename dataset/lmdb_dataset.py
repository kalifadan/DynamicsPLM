import abc
import torch
import lmdb
import pytorch_lightning as pl
import copy
import pickle
import pandas as pd
import numpy as np
import json
import os
import re

from torch.utils.data import DataLoader
from tqdm import tqdm
from Bio.Align import PairwiseAligner
import requests


_10TB = 10995116277760


class LMDBDataset(pl.LightningDataModule):
    """
    Abstract class from which other datasets inherit. We use LMDB database for all subclasses.
    """
    def __init__(self,
                 train_lmdb: str = None,
                 valid_lmdb: str = None,
                 test_lmdb: str = None,
                 dataloader_kwargs: dict = None):
        """
        Args:
            train_lmdb: path to train lmdb
            valid_lmdb: path to valid lmdb
            test_lmdb: path to test lmdb
            dataloader_kwargs: kwargs for dataloader
        """
        super().__init__()
        self.train_lmdb = train_lmdb
        self.valid_lmdb = valid_lmdb
        self.test_lmdb = test_lmdb
        self.dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else {}

        # ligands_filepath = "LMDB/PDBBind/LP_PDBBind_edited.csv"
        # self.pdbbind_df = pd.read_csv(ligands_filepath, index_col=0)
        # self.pdbbind_df = self.pdbbind_df.dropna(subset=["smiles", "uniprot_id"])
        # self.pdbbind_df[["ic50", "ki", "kd"]] = self.pdbbind_df["kd/ki"].apply(lambda x: pd.Series(self.parse_kd_ki(x))).fillna(0)
        # self.pdbbind_df[["ic50", "ki", "kd"]] = self.pdbbind_df[["ic50", "ki", "kd"]].apply(self.log_normalize)
        #
        # valid_uniprot_ids = []
        # for filename in os.listdir("example"):
        #     if ".pdb" in filename:
        #         uniprot_id = filename.split("-")[1]
        #         valid_uniprot_ids.append(uniprot_id)
        # print("Number of valid unique proteins:", len(list(set(valid_uniprot_ids))))
        #
        # self.pdbbind_df = self.pdbbind_df[self.pdbbind_df["uniprot_id"].isin(valid_uniprot_ids)]
        # print("Dataset Size:", len(self.pdbbind_df))

        self.env = None
        self.operator = None

    def is_initialized(self):
        return self.env is not None
    
    def _init_lmdb(self, path):
        if self.env is not None:
            self._close_lmdb()
            
        # open lmdb
        self.env = lmdb.open(path, lock=False, map_size=_10TB)
        self.operator = self.env.begin()
    
    def _close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.operator = None
    
    def _cursor(self):
        return self.operator.cursor()

    def _get(self, key: str or int):
        value = self.operator.get(str(key).encode())

        if value is not None:
            value = value.decode()

        return value
     
    def _dataloader(self, stage):
        self.dataloader_kwargs["shuffle"] = True if stage == "train" else False
        lmdb_path = getattr(self, f"{stage}_lmdb")
        dataset = copy.copy(self)
        dataset._init_lmdb(lmdb_path)
        setattr(dataset, "stage", stage)
        
        return DataLoader(dataset, collate_fn=dataset.collate_fn, **self.dataloader_kwargs)
    
    def train_dataloader(self):
        dataloader = self._dataloader("train")
        return dataloader

    def test_dataloader(self):
        return self._dataloader("test")
    
    def val_dataloader(self):
        return self._dataloader("valid")
        
    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def collate_fn(self, batch):
        """
        Datasets should implement it as the function will be set when initializing Dataloader

        Returns:
            inputs: dict
            labels: dict
        """
        raise NotImplementedError

    def find_most_similar(self, target_id, target_seq, all_sequences):
        best_match = None
        best_score = -1

        if target_seq is None or len(target_seq) == 0:
            return best_match, best_score

        # aligner = PairwiseAligner()
        # aligner.mode = "local"

        aligner = PairwiseAligner(scoring="blastp")
        aligner.mode = "global"

        perfect_score = aligner.score(target_seq, target_seq)  # Alignment of target with itself

        for uniprot_id, seq in all_sequences.items():
            try:
                if uniprot_id == target_id:  # Skip self-comparison
                    return uniprot_id, 1.0

                # Compute alignment score
                alignments = aligner.score(target_seq, seq)

                if alignments > best_score:
                    best_score = alignments
                    best_match = uniprot_id
            except Exception:
                continue

        return best_match, best_score / perfect_score

    def fetch_uniprot_sequence(self, uniprot_id):
        try:
            if "-" in uniprot_id:
                uniprot_id = uniprot_id.split("-")[1]

            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            response = requests.get(url)

            if response.status_code == 200:
                fasta_data = response.text
                return "".join(fasta_data.split("\n")[1:])
            else:
                print(f"Failed to retrieve sequence for {uniprot_id}")
                return None
        except Exception:
            print(f"Got Exception while retrieving sequence for {uniprot_id}")
            return None

    def log_normalize(self, values):
        return np.log10(values + 1e-8)  # Adding small epsilon to avoid log(0)

    def parse_kd_ki(self, value):
        """Extract IC50, Ki, and Kd values from the 'kd/ki' column and convert to nM."""
        ic50, ki, kd = None, None, None  # Default values if not found

        if pd.isna(value):
            return ic50, ki, kd  # Return None values if missing

        # Regex pattern to capture IC50, Ki, or Kd values with their units (nM or µM)
        match = re.findall(r"(IC50|Ki|Kd)\s*=\s*(\d*\.?\d+)\s*([numµmM]+)", value, re.IGNORECASE)

        for label, num, unit in match:
            num = float(num)  # Convert number to float

            unit = unit.lower().replace("µ", "u")  # Convert µM to uM
            if unit == "um":  # Micromolar (µM/uM) to nanomolar (nM)
                num *= 1000
            elif unit == "mm":  # Millimolar (mM) to nanomolar (nM)
                num *= 1_000_000
            elif unit == "nm":  # Nanomolar (nM) stays the same
                pass
            else:
                print(f"Warning: Unrecognized unit '{unit}' in value '{value}'")

            # Assign the value based on the type
            if label == "IC50":
                ic50 = num
            elif label == "Ki":
                ki = num
            elif label == "Kd":
                kd = num

        return ic50, ki, kd
