import torch
import os
import torchmetrics
import random

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers import AutoTokenizer


@register_model
class SaprotGeneratorModel(SaprotBaseModel):
    def __init__(self, **kwargs):
        super().__init__(task='lm', **kwargs)

    def initialize_metrics(self, stage):
        return {}
    
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
        logits = logits.view(-1, logits.size(-1))

        ligands_embeddings, ligands_token_ids = self.process_ligands_with_smiles(ligands)
        ligands_token_ids = ligands_token_ids[0]['input_ids']
        ligands_embeddings = ligands_embeddings.squeeze(0)
        fused_representation = outputs['hidden_states']

        generated_embeddings = self.ligand_generator(fused_representation)

        cosine_sim = torch.nn.functional.cosine_similarity(
            generated_embeddings.mean(dim=1), ligands_embeddings.mean(dim=1), dim=-1
        )
        generator_loss = 1 - cosine_sim.mean()

        remapped_token_ids = []
        for seq in ligands_token_ids:
            new_seq = []
            for t in seq:
                token_id = t.item()
                if token_id in self.ligand_tokenizer.full_to_allowed_id:
                    new_seq.append(self.ligand_tokenizer.full_to_allowed_id[token_id])
                else:
                    new_seq.append(self.ligand_tokenizer.full_to_allowed_id[self.ligand_tokenizer.pad_token_id])
            remapped_token_ids.append(torch.tensor(new_seq, device=ligands_token_ids.device))
        remapped_token_ids = torch.stack(remapped_token_ids, dim=0)

        smiles_logits = self.ligand_decoder(generated_embeddings, remapped_token_ids[:, :-1])
        smiles_loss = torch.nn.functional.cross_entropy(
            smiles_logits.view(-1, smiles_logits.size(-1)),
            remapped_token_ids[:, 1:].reshape(-1),
            ignore_index=self.ligand_tokenizer.token_to_id["[PAD]"]
        )

        # if (stage == 'train' and random.random() < 0.01) or (stage != 'train' and random.random() < 0.1):
        if stage != 'train':
            predicted_token_ids = self.ligand_decoder.generate(generated_embeddings)
            real_smiles = self.ligand_tokenizer.decode_remapped(remapped_token_ids[0], skip_special_tokens=True)
            predicted_smiles = self.ligand_tokenizer.decode_remapped(predicted_token_ids[0], skip_special_tokens=True)
            print(f"Real SMILES: {real_smiles} --- Predicted SMILES: {predicted_smiles} --- Ligands:{ligands}")

        total_loss = generator_loss + smiles_loss

        if stage == 'train':
            log_dict = {}
            log_dict["train_loss"] = total_loss
            log_dict["train_generator_loss"] = generator_loss
            log_dict["train_smiles_loss"] = smiles_loss
            self.log_info(log_dict)
            self.reset_metrics("train")

        return total_loss
    
    def test_epoch_end(self, outputs):
        log_dict = {}       # self.get_log_dict("test")
        log_dict["test_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()
        
        print(log_dict)
        self.log_info(log_dict)
        
        self.reset_metrics("test")
    
    def validation_epoch_end(self, outputs):
        log_dict = {}           # self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()
        
        self.log_info(log_dict)
        self.reset_metrics("valid")

        save_info = {
            "ligand_generator": self.ligand_generator.state_dict(),
            "ligand_decoder": self.ligand_decoder.state_dict(),
            "ligand_proj": self.ligand_proj.state_dict(),
            "default_ligand": self.default_ligand,
        }

        self.check_save_condition(log_dict["valid_loss"], mode="min", save_info=save_info)
