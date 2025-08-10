import torch
import json
import random
import numpy as np

from ..data_interface import register_dataset
from transformers import EsmTokenizer
from ..lmdb_dataset import *
from scipy.spatial.distance import pdist, squareform


def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
	batch_size = len(sequences)
	shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

	if dtype is None:
		dtype = sequences[0].dtype

	if isinstance(sequences[0], np.ndarray):
		array = np.full(shape, constant_value, dtype=dtype)
	elif isinstance(sequences[0], torch.Tensor):
		device = sequences[0].device
		array = torch.full(shape, constant_value, dtype=dtype, device=device)

	for arr, seq in zip(array, sequences):
		arrslice = tuple(slice(dim) for dim in seq.shape)
		arr[arrslice] = seq

	return array



@register_dataset
class DynamicPLMContactDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            max_length: Max length of sequence
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

        self.last_entry = None

    def __getitem__(self, index):
        try:
            entry = json.loads(self._get(index))
            if entry:
                self.last_entry = entry
        except Exception as e:
            print(f"Skipping index {index}: {e}")
            entry = self.last_entry if self.last_entry else json.loads(self._get(random.randint(0, len(self) - 1)))

        seq = entry['seq']
        dynamic_features = {}

        if "shp" in entry:
            dynamic_features["shp"] = entry["shp"]
        else:
            print("Found empty shp values for protein: ", entry["name"])

        tokens = self.tokenizer.tokenize(seq)[:self.max_length]
        seq = " ".join(tokens)

        valid_mask = np.array(entry['valid_mask'])[:self.max_length]
        coords = np.array(entry['tertiary'])[:self.max_length]
        contact_map = np.less(squareform(pdist(coords)), 8.0).astype(np.int64)

        y_inds, x_inds = np.indices(contact_map.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(y_inds - x_inds) < 6
        contact_map[invalid_mask] = -1

        return seq, contact_map, len(contact_map), dynamic_features

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, contact_maps, lengths, dynamic_features = tuple(zip(*batch))

        encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": encoder_info}

        contact_maps = pad_sequences(contact_maps, -1)
        targets = torch.tensor(contact_maps, dtype=torch.long)
        labels = {"targets": targets, "lengths": lengths}

        return inputs, labels, dynamic_features[0], None
