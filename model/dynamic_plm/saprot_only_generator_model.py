import torch
import os
import torchmetrics

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
from transformers.modeling_outputs import MaskedLMOutput


@register_model
class SaprotGeneratorModel(SaprotBaseModel):
    def __init__(self, **kwargs):
        super().__init__(task='lm', **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(ignore_index=-1)}
    
    def forward(self, inputs, coords=None, ligands=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        output = self.model.esm(**inputs)
        hidden = output[0]

        logits = self.model.lm_head(hidden)

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

        # Ligand loss
        ligands_embeddings, ligands_labels = self.process_ligands(ligands)
        ligands_embeddings = ligands_embeddings.squeeze(0)
        fused_representation = outputs['hidden_states']

        generated_embeddings = self.ligand_generator(fused_representation)
        cosine_sim = torch.nn.functional.cosine_similarity(generated_embeddings.unsqueeze(0),
                                                           ligands_embeddings.unsqueeze(1),
                                                           dim=-1)
        generator_loss = 1 - cosine_sim.mean()      # Compute loss: 1 - mean cosine similarity
        loss = generator_loss

        if stage == 'train':
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            log_dict["train_ligand_generator_loss"] = generator_loss
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
            "ligand_generator": self.ligand_generator.state_dict(),
            "ligand_proj": self.ligand_proj.state_dict(),
            "default_ligand": self.default_ligand,
        }

        self.check_save_condition(log_dict["valid_loss"], mode="min", save_info=save_info)
