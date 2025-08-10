import torch.distributed as dist
import os
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch.nn.functional import cross_entropy, cosine_similarity
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM

from ..model_interface import register_model
from .base import SaprotBaseModel, SHPEmbeddingLayer, SHPWrapper


@register_model
class DynamicPLMRegressionModel(SaprotBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            **kwargs: other arguments for SaprotBaseModel
        """
        super().__init__(task="regression", **kwargs)
        self.test_result_path = test_result_path

        # SHP Embedding layer
        pretrained_weights = self.model.esm.embeddings.word_embeddings.weight.data.clone()
        vocab_path = f"{self.config_path}/vocab.txt"

        # Initialize SHP embedding
        self.shp_embedding_layer = SHPEmbeddingLayer(vocab_file=vocab_path,
                                                     pretrained_embedding=pretrained_weights)

        # Register temp buffer for SHP
        self.current_shp_tensor = None  # This will be set in forward()

        self.shp_wrapper = SHPWrapper(self.shp_embedding_layer)
        self.model.esm.embeddings.word_embeddings = self.shp_wrapper

    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}

    def forward(self, inputs, dynamic_features=None):
        if self.freeze_backbone:
            # To be implemented
            raise NotImplementedError

        if dynamic_features is not None and "shp" in dynamic_features:
            shp_logits = torch.tensor(dynamic_features["shp"], dtype=torch.float32, device=inputs["input_ids"].device)
            shp = F.softmax(shp_logits, dim=-1)
            self.shp_wrapper.set_shp_tensor(shp)

        else:
            self.shp_wrapper.set_shp_tensor(None)

        return self.model(**inputs).logits.squeeze(-1)

        # if "shp" not in dynamic_features:
        #     return self.model(**inputs).logits.squeeze(dim=-1)
        #
        # output = self.model.esm(**inputs)
        # hidden = output[0]
        #
        # shp_features = torch.tensor(dynamic_features["shp"]).squeeze()
        # # shp = softmax(dynamics_pred["shp"].squeeze(), dim=1).cpu().numpy()
        # # print("shp_features:", shp_features.shape)
        # shp_embeddings = self.shp_proj(shp_features)
        # print("shp_embeddings:", shp_embeddings.shape)
        # # #
        # # # # Apply cross-attention (Protein as Query, SHP as Key/Value)
        # # # attn_output, _ = self.cross_attention(
        # # #     query=hidden,  # Protein embeddings
        # # #     key=shp_embeddings,
        # # #     value=shp_embeddings
        # # # )
        # # # # attn_output = self.norm(attn_output)  # Optional normalization before addition
        # # # hidden = hidden + attn_output  # Residual connection
        # #
        # logits = self.model.classifier(hidden).squeeze(dim=-1)
        #
        # return logits

    def loss_func(self, stage, outputs, labels, inputs=None, info=None):
        fitness = labels['labels'].to(outputs)
        task_loss = torch.nn.functional.mse_loss(outputs, fitness)

        if stage == "test" and self.test_result_path is not None:
            os.makedirs(os.path.dirname(self.test_result_path), exist_ok=True)
            with open(self.test_result_path, 'a') as w:
                uniprot_id, protein_type = info[0]
                w.write(f"{uniprot_id}\t{protein_type}\t{outputs.detach().float().item()}\t{fitness.float().item()}\n")

        loss = task_loss
        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.update(outputs.detach().float(), fitness.float())

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["loss"] = loss
            log_dict["task_loss"] = task_loss
            log_dict["avg_gate"] = self.shp_embedding_layer.avg_gate
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def test_epoch_end(self, outputs):
        log_dict = self.get_log_dict("test")
        print(log_dict)

        self.log_info(log_dict)
        self.reset_metrics("test")

    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
