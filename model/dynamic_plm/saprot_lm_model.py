import torch
import os
import torchmetrics

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
from transformers.modeling_outputs import MaskedLMOutput


@register_model
class SaprotLMModel(SaprotBaseModel):
    def __init__(self, **kwargs):
        super().__init__(task='lm', **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(ignore_index=-1)}
    
    def forward(self, inputs, coords=None, ligands=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        # outputs = self.model(**inputs)
        output = self.model.esm(**inputs)
        hidden = output[0]

        # Apply cross-attention (Protein as Query, Ligands as Key/Value)
        ligands_embeddings, ligands_labels = self.process_ligands(ligands)
        ligands_embeddings = ligands_embeddings.squeeze(0)
        ligands_embeddings = self.ligand_proj(ligands_embeddings)  # [batch, ligand_dim] → [batch, hidden_dim]
        ligands_embeddings = ligands_embeddings.unsqueeze(1).expand(-1, hidden.size(1), -1)  # Expand for attention
        attn_output, _ = self.cross_attention(
            query=hidden,
            key=ligands_embeddings,
            value=ligands_embeddings
        )
        fused_representation = hidden + attn_output  # Residual connection
        logits = self.model.lm_head(fused_representation)

        outputs = MaskedLMOutput(
            logits=logits,
            hidden_states=hidden,
            attentions=output.attentions,
        )
        
        return outputs
    
    def loss_func(self, stage, outputs, labels, inputs=None, ligands=None, info=None):
        logits = outputs['logits']

        # merge the first and second dimension of logits
        logits = logits.view(-1, logits.size(-1))
        
        # flatten labels
        labels = labels['labels'].flatten().to(logits.device)

        loss = cross_entropy(logits, labels, ignore_index=-1)
        getattr(self, f"{stage}_acc").update(logits.detach(), labels)

        # Ligand loss
        ligands_embeddings, ligands_labels = self.process_ligands(ligands)
        ligands_labels = torch.tensor(ligands_labels, dtype=torch.float32).to(self.model.device)
        fused_representation = outputs['hidden_states']

        task_loss = loss

        if stage == 'train':
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            log_dict["train_mlm_loss"] = task_loss
            # log_dict["train_mse_ba_loss"] = ba_loss
            self.log_info(log_dict)
            self.reset_metrics("train")
        
        return loss
    
    def test_epoch_end(self, outputs):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()
        
        print(log_dict)
        self.log_info(log_dict)
        
        self.reset_metrics("test")
    
    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()
        
        self.log_info(log_dict)
        self.reset_metrics("valid")

        save_info = {
            "cross_attention": self.cross_attention.state_dict(),
            "ligand_proj": self.ligand_proj.state_dict(),
            "default_ligand": self.default_ligand,  # Directly saving nn.Parameter
        }

        self.check_save_condition(log_dict["valid_loss"], mode="min", save_info=save_info)
