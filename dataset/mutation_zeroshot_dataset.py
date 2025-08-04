import json
import torch

from .data_interface import register_dataset
from .lmdb_dataset import LMDBDataset


@register_dataset
class MutationZeroShotDataset(LMDBDataset):
    """
    Dataset that deals with mutation data for zero-shot prediction
    """

    def __init__(self, **kwargs):
        """

        Args: **kwargs: other arguments for LMDBDataset

        """
        super().__init__(**kwargs)

        self.proteins_without_ligands = []
        self.proteins_with_ligands_ids = []
        self.proteins_with_ligands_indexes = []

    def __getitem__(self, index):
        data = json.loads(self._get(index))

        # Ligands Extraction
        uniprot_seq, ligand_list = data['seq'], []
        ligand_list = self.pdbbind_df[self.pdbbind_df['seq'] == uniprot_seq][["smiles", "value", "ic50",
                                                                              "kd", "ki"]].values.tolist()
        ligand_list = [(item[0], [item[1], item[2], item[3], item[4]]) for item in ligand_list]

        print("ligand_list", ligand_list)

        return data["seq"], data["mut_info"], data["fitness"], ligand_list, None

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, mut_info, fitness, ligand_list, info = zip(*batch)
        
        plddt = self._get("plddt")
        if plddt is not None:
            plddt = json.loads(plddt)
        
        inputs = {"wild_type": self._get("wild_type"),
                  "seqs": seqs,
                  "mut_info": mut_info,
                  "structure_content": self._get("structure_content"),
                  "structure_type": self._get("structure_type"),
                  "plddt": plddt}

        labels = {"labels": torch.Tensor(fitness)}

        return inputs, labels, ligand_list, info
