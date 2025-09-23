import numpy as np
import torchmetrics
import torch
import os

from torch.nn import Linear, ReLU, Sequential, Sigmoid
import torch.nn.functional as F
from torch.nn.functional import cross_entropy, cosine_similarity
from ..model_interface import register_model
from .base import SaprotBaseModel, SHPEmbeddingLayer, SHPWrapper


@register_model
class DynamicPLMPPIModel(SaprotBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        """
        Args:
            **kwargs: other arguments for SaprotBaseModel
        """
        super().__init__(task="base", **kwargs)
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

    def forward(self, inputs_1, inputs_2, dynamic_features=None):
        if self.freeze_backbone:
            # To be implemented
            raise NotImplementedError

        if dynamic_features is not None and "shp_1" in dynamic_features:
            shp_logits = torch.tensor(dynamic_features["shp_1"], dtype=torch.float32, device=inputs_1["input_ids"].device)
            shp = F.softmax(shp_logits, dim=-1)
            self.shp_wrapper.set_shp_tensor(shp)
        else:
            self.shp_wrapper.set_shp_tensor(None)

        hidden_1 = self.model.esm(**inputs_1)[0][:, 0, :]

        if dynamic_features is not None and "shp_2" in dynamic_features:
            shp_logits = torch.tensor(dynamic_features["shp_2"], dtype=torch.float32, device=inputs_2["input_ids"].device)
            shp = F.softmax(shp_logits, dim=-1)
            self.shp_wrapper.set_shp_tensor(shp)
        else:
            self.shp_wrapper.set_shp_tensor(None)

        hidden_2 = self.model.esm(**inputs_2)[0][:, 0, :]

        hidden_concat = torch.cat([hidden_1, hidden_2], dim=-1)
        return self.model.classifier(hidden_concat)
    
    def loss_func(self, stage, logits, labels, inputs=None, info=None):
        label = labels['labels']
        task_loss = cross_entropy(logits, label)
        loss = task_loss

        if stage == "test" and self.test_result_path is not None:
            os.makedirs(os.path.dirname(self.test_result_path), exist_ok=True)
            with open(self.test_result_path, 'a') as w:
                uniprot_id_1 = info["protein_1"][0]
                uniprot_id_2 = info["protein_2"][0]
                probs = F.softmax(logits, dim=1).squeeze().tolist()
                probs_str = "\t".join([f"{p:.4f}" for p in probs])
                w.write(f"{uniprot_id_1}\t{uniprot_id_2}\t{probs_str}\t{label.item()}\n")

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
            log_dict["avg_gate"] = self.shp_embedding_layer.avg_gate
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
