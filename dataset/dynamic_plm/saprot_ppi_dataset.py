import torch
import json
import pickle
import random

from ..lmdb_dataset import LMDBDataset
from transformers import EsmConfig, EsmTokenizer
from ..data_interface import register_dataset


@register_dataset
class SaprotPPIDataset(LMDBDataset):
    def __init__(self,
             tokenizer: str,
             max_length: int = 1024,
             plddt_threshold: float = None,
             **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            
            max_length: Max length of sequence
            
            plddt_threshold: If not None, mask structure tokens with pLDDT < threshold
            
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.plddt_threshold = plddt_threshold

        self.proteins_without_ligands = []
        self.proteins_with_ligands_ids = []
        self.proteins_with_ligands_indexes = []

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq_1, seq_2 = entry['seq_1'], entry['seq_2']

        # Ligands Extraction
        uniprot_id_1, ligand_list_1 = entry['name_1'], []
        ligand_list_1 = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id_1][["smiles", "value", "ic50",
                                                                                        "kd", "ki", "type"]].values.tolist()
        protein_type = [item[5] for item in ligand_list_1]
        protein_type = "Unknown" if len(protein_type) == 0 else protein_type[0]
        ligand_list_1 = [(item[0], [item[1], item[2], item[3], item[4]]) for item in ligand_list_1]
        information_list_1 = [uniprot_id_1, protein_type]

        uniprot_id_2, ligand_list_2 = entry['name_2'], []
        ligand_list_2 = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id_2][["smiles", "value", "ic50",
                                                                                        "kd", "ki", "type"]].values.tolist()
        protein_type = [item[5] for item in ligand_list_2]
        protein_type = "Unknown" if len(protein_type) == 0 else protein_type[0]
        ligand_list_2 = [(item[0], [item[1], item[2], item[3], item[4]]) for item in ligand_list_2]
        information_list_2 = [uniprot_id_2, protein_type]

        # Mask structure tokens with pLDDT < threshold
        if self.plddt_threshold is not None:
            plddt_1, plddt_2 = entry['plddt_1'], entry['plddt_2']
            tokens = self.tokenizer.tokenize(seq_1)
            seq_1 = ""
            for token, score in zip(tokens, plddt_1):
                if score < self.plddt_threshold:
                    seq_1 += token[:-1] + "#"
                else:
                    seq_1 += token

            tokens = self.tokenizer.tokenize(seq_2)
            seq_2 = ""
            for token, score in zip(tokens, plddt_2):
                if score < self.plddt_threshold:
                    seq_2 += token[:-1] + "#"
                else:
                    seq_2 += token

        tokens = self.tokenizer.tokenize(seq_1)[:self.max_length]
        seq_1 = " ".join(tokens)

        tokens = self.tokenizer.tokenize(seq_2)[:self.max_length]
        seq_2 = " ".join(tokens)

        return seq_1, seq_2, int(entry["label"]), ligand_list_1, ligand_list_2, information_list_1, information_list_2

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs_1, seqs_2, label_ids, ligand_list_1, ligand_list_2, information_list_1, information_list_2 = tuple(zip(*batch))

        label_ids = torch.tensor(label_ids, dtype=torch.long)
        labels = {"labels": label_ids}

        encoder_info_1 = self.tokenizer.batch_encode_plus(seqs_1, return_tensors='pt', padding=True)
        encoder_info_2 = self.tokenizer.batch_encode_plus(seqs_2, return_tensors='pt', padding=True)
        inputs = {"inputs_1": encoder_info_1,
                  "inputs_2": encoder_info_2}

        ligands = {"ligands_1": ligand_list_1,
                   "ligands_2": ligand_list_2}

        info = {"protein_1": information_list_1,
                "protein_2": information_list_2}

        return inputs, labels, ligands, info
