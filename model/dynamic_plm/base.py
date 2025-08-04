import torch
import torch.nn as nn

from typing import List, Dict
# from data.pdb2feature import batch_coords2feature
from transformers import EsmConfig, EsmTokenizer, EsmForMaskedLM, EsmForSequenceClassification
# from module.esm.structure_module import (
#     EsmForMaskedLMWithStructure as EsmForMaskedLM,
#     EsmForSequenceClassificationWithStructure as EsmForSequenceClassification,
# )
from ..abstract_model import AbstractModel


class SHPWrapper(nn.Module):
    def __init__(self, embedding_layer: nn.Module):
        super().__init__()
        self.embedding_layer = embedding_layer
        self._shp_tensor = None  # can be None to disable SHP

    def set_shp_tensor(self, shp_tensor: torch.Tensor | None):
        self._shp_tensor = shp_tensor

    def forward(self, input_ids: torch.Tensor):
        return self.embedding_layer(input_ids, self._shp_tensor)


class SHPEmbeddingLayer(nn.Module):
    def __init__(self, vocab_file: str, pretrained_embedding: torch.Tensor):
        super().__init__()

        with open(vocab_file, "r") as f:
            tokens = [line.strip() for line in f.readlines()]

        self.token_to_index = {tok: idx for idx, tok in enumerate(tokens)}
        self.index_to_token = {idx: tok for tok, idx in self.token_to_index.items()}
        self.special_tokens = {"<cls>", "<pad>", "<eos>", "<unk>", "<mask>"}
        self.special_token_ids = {self.token_to_index[tok] for tok in self.special_tokens if tok in self.token_to_index}

        self.seq_vocab = []
        self.struct_vocab = []

        for tok in tokens:
            if tok in self.special_tokens or len(tok) != 2:
                continue
            seq_tok, struct_tok = tok
            if seq_tok not in self.seq_vocab:
                self.seq_vocab.append(seq_tok)
            if struct_tok != "#" and struct_tok not in self.struct_vocab:
                self.struct_vocab.append(struct_tok)

        self.seq_vocab_map = {tok: i for i, tok in enumerate(self.seq_vocab)}
        self.struct_vocab_map = {tok: i for i, tok in enumerate(self.struct_vocab)}

        self.seq_vocab_size = len(self.seq_vocab)
        self.struct_vocab_size = len(self.struct_vocab)
        self.hidden_dim = pretrained_embedding.shape[1]

        # Special tokens + masked structure tokens (like A#)
        self.special_index_to_position = {}
        self.special_embedding = nn.Parameter(torch.zeros(len(self.special_token_ids) + 21, self.hidden_dim))

        # Embedding table: shape (21, 20, D)
        self.full_embed = nn.Parameter(torch.zeros(self.seq_vocab_size, self.struct_vocab_size, self.hidden_dim))

        # Load weights from pretrained embedding
        for idx, tok in self.index_to_token.items():
            if tok in self.special_tokens or (len(tok) == 2 and tok[1] == "#"):
                if idx not in self.special_index_to_position:
                    pos = len(self.special_index_to_position)
                    self.special_index_to_position[idx] = pos
                    self.special_embedding.data[pos] = pretrained_embedding[idx]
            elif len(tok) == 2:
                seq_tok, struct_tok = tok
                if seq_tok in self.seq_vocab_map and struct_tok in self.struct_vocab_map:
                    i = self.seq_vocab_map[seq_tok]
                    j = self.struct_vocab_map[struct_tok]
                    self.full_embed.data[i, j] = pretrained_embedding[idx]

        # Gating network: learn how much to use SHP vs token
        # self.gate_net = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, 1),
        #     nn.Sigmoid()
        # )
        # self.gate_net[2].bias.data.fill_(-0.85)   # sigmoid(-0.85) ≈ 0.3

        self.gate_net = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),  # Normalize input
            nn.Linear(self.hidden_dim, self.hidden_dim),  # First projection
            nn.GELU(),  # Smoother non-linearity
            nn.Dropout(0.1),  # Add regularization
            nn.Linear(self.hidden_dim, 1),  # Scalar gate output
            nn.Sigmoid()  # Output in (0,1)
        )
        self.gate_net[4].bias.data.fill_(-0.85)

        # self.cross_attention = nn.MultiheadAttention(
        #     embed_dim=self.hidden_dim,
        #     num_heads=4,
        #     batch_first=True
        # )

        self.avg_gate = 0

    def embed_token_by_id(self, token_id: int):
        token = self.index_to_token[token_id]
        if token_id in self.special_index_to_position:
            return self.special_embedding[self.special_index_to_position[token_id]]
        elif len(token) == 2:
            seq_tok, struct_tok = token
            if struct_tok == "#":
                return self.special_embedding[self.special_index_to_position[token_id]]
            return self.full_embed[self.seq_vocab_map[seq_tok], self.struct_vocab_map[struct_tok]]
        else:
            raise ValueError(f"Unexpected token: {token}")

    def forward(self, input_ids: torch.Tensor, shp_tensor: torch.Tensor | None = None):
        B, L = input_ids.shape
        D = self.hidden_dim
        device = input_ids.device
        output = torch.zeros((B, L, D), device=device)

        # Fast fallback mode (no SHP): just use pretrained embeddings
        if shp_tensor is None:
            print("Got None SHP vector!")
            flat_input = input_ids.view(-1)
            flat_embed = torch.stack([self.embed_token_by_id(tok.item()) for tok in flat_input])
            return flat_embed.view(B, L, D)

        # Handle special tokens
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_positions = torch.zeros_like(input_ids)

        for i in range(B):
            for j in range(L):
                token_id = input_ids[i, j].item()
                if token_id in self.special_index_to_position:
                    special_mask[i, j] = True
                    special_positions[i, j] = self.special_index_to_position[token_id]

        if special_mask.any():
            output[special_mask] = self.special_embedding[special_positions[special_mask]]

        # Regular tokens
        shp_len = shp_tensor.shape[1]
        max_shp_len = min(shp_len, L - 2)  # skip CLS/EOS
        if max_shp_len <= 0:
            return output

        reg_i_list, reg_j_list, seq_ids = [], [], []
        shp_vals = []
        E_token_list = []

        for b in range(B):
            for offset in range(max_shp_len):
                j = offset + 1
                token_id = input_ids[b, j].item()
                token = self.index_to_token.get(token_id, None)
                if not token or len(token) != 2 or token_id in self.special_index_to_position:
                    continue
                seq_tok = token[0]
                if seq_tok not in self.seq_vocab_map:
                    continue
                reg_i_list.append(b)
                reg_j_list.append(j)
                seq_ids.append(self.seq_vocab_map[seq_tok])
                shp_vals.append(shp_tensor[b, offset])
                E_token_list.append(self.embed_token_by_id(token_id))

        if not reg_i_list:
            return output

        reg_i = torch.tensor(reg_i_list, device=device)
        reg_j = torch.tensor(reg_j_list, device=device)
        seq_ids = torch.tensor(seq_ids, device=device)
        shp_vals = torch.stack(shp_vals).to(device)  # (N, 20)
        E_token = torch.stack(E_token_list).to(device)  # (N, D)

        # Weighted structure embedding
        struct_embeds = self.full_embed[seq_ids]     # (N, 20, D)
        shp_vals = shp_vals.unsqueeze(-1)            # (N, 20, 1)
        E_shp = (struct_embeds * shp_vals).sum(dim=1)  # (N, D)

        # Learn fusion via gate
        gate = self.gate_net(E_token)  # (N, 1)
        E_final = gate * E_shp + (1 - gate) * E_token  # (N, D)
        # E_final = 0.5 * E_shp + 0.5 * E_token  # (N, D)

        if self.training:  # Only log during training
            self.avg_gate = gate.mean().item()

        # SHP values used to modulate VALUE — not key/query
        # value = struct_embeds * shp_vals.unsqueeze(-1)  # (N, 20, D)
        # key = struct_embeds  # (N, 20, D)
        # query = E_token.unsqueeze(1)  # (N, 1, D)
        #
        # # Run cross-attention from E_token to structure conformations
        # E_shp_attn, _ = self.cross_attention(query, key, value)  # (N, 1, D)
        # E_final = E_shp_attn.squeeze(1) + E_token  # residual

        output[reg_i, reg_j] = E_final
        return output


class SaprotBaseModel(AbstractModel):
    """
    ESM base model. It cannot be used directly but provides model initialization for downstream tasks.
    """

    def __init__(self,
                 task: str,
                 config_path: str,
                 extra_config: dict = None,
                 load_pretrained: bool = False,
                 freeze_backbone: bool = False,
                 use_lora: bool = False,
                 lora_config_path: str = None,
                 **kwargs):
        """
        Args:
            task: Task name. Must be one of ['classification', 'regression', 'lm', 'base']

            config_path: Path to the config file of huggingface esm model

            extra_config: Extra config for the model

            load_pretrained: Whether to load pretrained weights of base model

            freeze_backbone: Whether to freeze the backbone of the model

            use_lora: Whether to use LoRA on downstream tasks

            lora_config_path: Path to the config file of LoRA. If not None, LoRA model is for inference only.
            Otherwise, LoRA model is for training.

            **kwargs: Other arguments for AbstractModel
        """
        assert task in ['classification', 'regression', 'lm', 'base']
        self.task = task
        self.config_path = config_path
        self.extra_config = extra_config
        self.load_pretrained = load_pretrained
        self.freeze_backbone = freeze_backbone
        super().__init__(**kwargs)

        # After all initialization done, lora technique is applied if needed
        self.use_lora = use_lora
        if use_lora:
            self._init_lora(lora_config_path)

    def _init_lora(self, lora_config_path):
        from peft import (
            PeftModelForSequenceClassification,
            get_peft_model,
            LoraConfig,
        )

        if lora_config_path:
            # Note that the model is for inference only
            self.model = PeftModelForSequenceClassification.from_pretrained(self.model, lora_config_path)
            self.model.merge_and_unload()
            print("LoRA model is initialized for inference.")

        else:
            lora_config = {
                "task_type": "SEQ_CLS",
                "target_modules": ["query", "key", "value", "intermediate.dense", "output.dense"],
                "modules_to_save": ["classifier"],
                "inference_mode": False,
                "lora_dropout": 0.1,
                "lora_alpha": 8,
            }

            peft_config = LoraConfig(**lora_config)
            self.model = get_peft_model(self.model, peft_config)
            # original_module is not needed for training
            self.model.classifier.original_module = None

            print("LoRA model is initialized for training.")
            self.model.print_trainable_parameters()

        # After LoRA model is initialized, add trainable parameters to optimizer
        self.init_optimizers()

    def initialize_model(self):
        # Initialize tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(self.config_path)

        # Initialize different models according to task
        config = EsmConfig.from_pretrained(self.config_path)

        # Add extra config if needed
        if self.extra_config is None:
            self.extra_config = {}

        for k, v in self.extra_config.items():
            setattr(config, k, v)

        if self.task == 'classification':
            # Note that self.num_labels should be set in child classes
            if self.load_pretrained:
                self.model = EsmForSequenceClassification.from_pretrained(
                    self.config_path, num_labels=self.num_labels, **self.extra_config)

            else:
                config.num_labels = self.num_labels
                self.model = EsmForSequenceClassification(config)

        elif self.task == 'regression':
            if self.load_pretrained:
                self.model = EsmForSequenceClassification.from_pretrained(
                    self.config_path, num_labels=1, **self.extra_config)

            else:
                config.num_labels = 1
                self.model = EsmForSequenceClassification(config)

        elif self.task == 'lm':
            if self.load_pretrained:
                self.model = EsmForMaskedLM.from_pretrained(self.config_path, **self.extra_config)

            else:
                self.model = EsmForMaskedLM(config)

        elif self.task == 'base':
            if self.load_pretrained:
                self.model = EsmForMaskedLM.from_pretrained(self.config_path, **self.extra_config)

            else:
                self.model = EsmForMaskedLM(config)

            # Remove lm_head as it is not needed for PPI task
            self.model.lm_head = None

        # Freeze the backbone of the model
        if self.freeze_backbone:
            for param in self.model.esm.parameters():
                param.requires_grad = False

    def initialize_metrics(self, stage: str) -> dict:
        return {}

    def get_hidden_states(self, inputs, reduction: str = None) -> list:
        """
        Get hidden representations of the model.

        Args:
            inputs:  A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
            reduction: Whether to reduce the hidden states. If None, the hidden states are not reduced. If "mean",
                        the hidden states are averaged over the sequence length.

        Returns:
            hidden_states: A list of tensors. Each tensor is of shape [L, D], where L is the sequence length and D is
                            the hidden dimension.
        """
        inputs["output_hidden_states"] = True
        outputs = self.model.esm(**inputs)

        # Get the index of the first <eos> token
        input_ids = inputs["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        ends = (input_ids == eos_id).int()
        indices = ends.argmax(dim=-1)

        repr_list = []
        hidden_states = outputs["hidden_states"][-1]
        for i, idx in enumerate(indices):
            if reduction == "mean":
                repr = hidden_states[i][1:idx].mean(dim=0)
            else:
                repr = hidden_states[i][1:idx]

            repr_list.append(repr)

        return repr_list

    def get_protein_representation(self, inputs, reduction: str = None):
        """
        Get hidden representation of a protein.

        Args:
            inputs:  A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
            reduction: Whether to reduce the hidden states. If None, the hidden states are not reduced. If "mean",
                        the hidden states are averaged over the sequence length.

        Returns:
            hidden: A tensor. Each tensor is of shape [L, D], where L is the sequence length and D is
                            the hidden dimension.
        """
        inputs["output_hidden_states"] = True
        outputs = self.model.esm(**inputs)
        hidden = outputs[0]
        ligands_embeddings = self.ligand_generator(hidden[:, 0, :])
        ligands_embeddings = self.ligand_proj(ligands_embeddings)  # [batch, ligand_dim] → [batch, hidden_dim]
        ligands_embeddings = ligands_embeddings.unsqueeze(1).expand(-1, hidden.size(1), -1)  # Expand for attention

        # Apply cross-attention (Protein as Query, Ligands as Key/Value)
        attn_output, _ = self.cross_attention(
            query=hidden,  # Protein embeddings
            key=ligands_embeddings,
            value=ligands_embeddings
        )

        hidden = hidden + attn_output  # Residual connection
        return hidden

    # def add_bias_feature(self, inputs, coords: List[Dict]) -> torch.Tensor:
    #     """
    #     Add structure information as biases to attention map. This function is used to add structure information
    #     to the model as Evoformer does.
    #
    #     Args:
    #         inputs: A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
    #         coords: Coordinates of backbone atoms. Each element is a dictionary with keys ["N", "CA", "C", "O"].
    #
    #     Returns
    #         pair_feature: A tensor of shape [B, L, L, 407]. Here 407 is the RBF of distance(400) + angle(7).
    #     """
    #     inputs["pair_feature"] = batch_coords2feature(coords, self.model.device)
    #     return inputs

    def save_checkpoint(self, save_info: dict = None) -> None:
        """
        Rewrite this function for saving LoRA parameters
        """
        if not self.use_lora:
            return super().save_checkpoint(save_info)

        else:
            self.model.save_pretrained(self.save_path)


