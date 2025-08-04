import numpy as np
import torchmetrics
import torch
import os

from torch.nn import Linear, ReLU, Sequential, Sigmoid
import torch.nn.functional as F
from torch.nn.functional import cross_entropy, cosine_similarity
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotPPIModel(SaprotBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        """
        Args:
            **kwargs: other arguments for SaprotBaseModel
        """
        super().__init__(task="base", **kwargs)
        self.test_result_path = test_result_path

    def initialize_model(self):
        super().initialize_model()
        
        hidden_size = self.model.config.hidden_size * 2
        classifier = torch.nn.Sequential(
                        Linear(hidden_size, hidden_size),
                        ReLU(),
                        Linear(hidden_size, 2)
                    )
        
        setattr(self.model, "classifier", classifier)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(),
                f"{stage}_auroc": torchmetrics.AUROC(task="binary")
               }

    def forward(self, inputs_1, inputs_2, ligands=None):
        if self.freeze_backbone:
            hidden_1 = torch.stack(self.get_hidden_states(inputs_1, reduction="mean"))
            hidden_2 = torch.stack(self.get_hidden_states(inputs_2, reduction="mean"))
        else:
            hidden_1 = self.model.esm(**inputs_1)[0][:, 0, :]
            hidden_2 = self.model.esm(**inputs_2)[0][:, 0, :]

        # if [] not in ligands['ligands_1']:
        #     ligands_embeddings_1, ligands_labels_1 = self.process_ligands(ligands['ligands_1'])
        #     ligands_embeddings_1 = ligands_embeddings_1.squeeze(0)
        # else:
        #     ligands_embeddings_1 = self.ligand_generator(hidden_1)
        # 
        # if [] not in ligands['ligands_2']:
        #     ligands_embeddings_2, ligands_labels_2 = self.process_ligands(ligands['ligands_2'])
        #     ligands_embeddings_2 = ligands_embeddings_2.squeeze(0)
        # else:
        #     ligands_embeddings_2 = self.ligand_generator(hidden_2)
        # 
        # # Apply cross-attention (Protein as Query, Ligands as Key/Value)
        # ligands_embeddings_1 = self.ligand_proj(ligands_embeddings_1)  # [batch, ligand_dim] → [batch, hidden_dim]
        # 
        # attn_output_1, _ = self.cross_attention(
        #     query=hidden_1,  # Protein embeddings
        #     key=ligands_embeddings_1,
        #     value=ligands_embeddings_1
        # )
        # hidden_1 = hidden_1 + attn_output_1  # Residual connection
        # 
        # # Apply cross-attention (Protein as Query, Ligands as Key/Value)
        # ligands_embeddings_2 = self.ligand_proj(ligands_embeddings_2)  # [batch, ligand_dim] → [batch, hidden_dim]
        # 
        # attn_output_2, _ = self.cross_attention(
        #     query=hidden_2,  # Protein embeddings
        #     key=ligands_embeddings_2,
        #     value=ligands_embeddings_2
        # )
        # hidden_2 = hidden_2 + attn_output_2  # Residual connection

        hidden_concat = torch.cat([hidden_1, hidden_2], dim=-1)
        return self.model.classifier(hidden_concat)
    
    def loss_func(self, stage, logits, labels, inputs=None, ligands=None, info=None):
        label = labels['labels']
        task_loss = cross_entropy(logits, label)
        loss = task_loss

        if stage == "test" and self.test_result_path is not None:
            os.makedirs(os.path.dirname(self.test_result_path), exist_ok=True)
            with open(self.test_result_path, 'a') as w:
                uniprot_id_1, protein_type_1 = info["protein_1"][0]
                uniprot_id_2, protein_type_2 = info["protein_2"][0]
                probs = F.softmax(logits, dim=1).squeeze().tolist()
                probs_str = "\t".join([f"{p:.4f}" for p in probs])
                w.write(f"{uniprot_id_1}\t{protein_type_1}\t{uniprot_id_2}\t{protein_type_2}\t{probs_str}\t{label.item()}\n")

        # Update metrics
        # for metric in self.metrics[stage].values():
        #     metric.update(logits.detach(), label)

        probs_pos = torch.softmax(logits, dim=1)[:, 1]  # shape [batch]

        for name, metric in self.metrics[stage].items():
            if "auroc" in name or "auprc" in name:
                metric.update(probs_pos.detach(), label)
            else:  # accuracy
                metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
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

        self.check_save_condition(log_dict["valid_acc"], mode="max")
        # self.check_save_condition(log_dict["valid_loss"], mode="min")
