import torchmetrics
import torch
import os

from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
from utils.metrics import count_f1_max
from ..model_interface import register_model
from .base import SaprotBaseModel, SHPEmbeddingLayer, SHPWrapper


@register_model
class DynamicPLMAnnotationModel(SaprotBaseModel):
    def __init__(self, anno_type: str, test_result_path: str = None, **kwargs):
        """
        Args:
            anno_type: one of EC, GO, GO_MF, GO_CC
            **kwargs: other parameters for SaprotBaseModel
        """
        label2num = {"EC": 585, "GO_BP": 1943, "GO_MF": 489, "GO_CC": 320}
        self.num_labels = label2num[anno_type]
        self.test_result_path = test_result_path
        super().__init__(task="classification", **kwargs)

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
        return {f"{stage}_aupr": torchmetrics.AveragePrecision(pos_label=1, average='micro')}

    def forward(self, inputs, dynamic_features=None, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        if self.freeze_backbone:
            # To be implemented
            raise NotImplementedError

        if dynamic_features is not None and "shp" in dynamic_features:
            shp_logits = torch.tensor(dynamic_features["shp"], dtype=torch.float32, device=inputs["input_ids"].device)
            shp = F.softmax(shp_logits, dim=-1)
            self.shp_wrapper.set_shp_tensor(shp)
        else:
            self.shp_wrapper.set_shp_tensor(None)

        return self.model(**inputs).logits

    def loss_func(self, stage, logits, labels, inputs=None, info=None):
        label = labels['labels'].to(logits)
        task_loss = binary_cross_entropy_with_logits(logits, label.float())
        aupr = getattr(self, f"{stage}_aupr")(logits.sigmoid().detach(), label)

        loss = task_loss

        if stage == "test" and self.test_result_path is not None:
            os.makedirs(os.path.dirname(self.test_result_path), exist_ok=True)
            with open(self.test_result_path, 'a') as w:
                for b in range(logits.size(0)):  # handle batch size > 1
                    uniprot_id = info[b]
                    probs = torch.sigmoid(logits[b]).tolist()
                    labels_vec = label[b].tolist()
                    # tab-separated strings
                    probs_str = "\t".join(f"{p:.4f}" for p in probs)
                    labels_str = "\t".join(str(int(l)) for l in labels_vec)
                    # id \t probs \t labels
                    w.write(f"{uniprot_id}\t{probs_str}\t{labels_str}\n")

        if stage == "train":
            log_dict = {"train_loss": loss}
            log_dict["task_loss"] = task_loss
            log_dict["avg_gate"] = self.shp_embedding_layer.avg_gate
            self.log_info(log_dict)
            self.reset_metrics("train")
        
        return loss
    
    def test_epoch_end(self, outputs):
        preds = self.all_gather(torch.cat(self.test_aupr.preds, dim=-1)).view(-1, self.num_labels)
        target = self.all_gather(torch.cat(self.test_aupr.target, dim=-1)).long().view(-1, self.num_labels)
        fmax = count_f1_max(preds, target)
        
        log_dict = {"test_f1_max": fmax,
                    "test_loss": torch.cat(self.all_gather(outputs), dim=-1).mean(),
                    # "test_aupr": self.test_aupr.compute()
                    }
        self.log_info(log_dict)
        print(log_dict)
        self.reset_metrics("test")

    def validation_epoch_end(self, outputs):
        aupr = self.valid_aupr.compute()

        preds = self.all_gather(torch.cat(self.valid_aupr.preds, dim=-1)).view(-1, self.num_labels)
        target = self.all_gather(torch.cat(self.valid_aupr.target, dim=-1)).long().view(-1, self.num_labels)
        f1_max = count_f1_max(preds, target)
        
        log_dict = {"valid_f1_max": f1_max,
                    "valid_loss": torch.cat(self.all_gather(outputs), dim=-1).mean(),
                    # "valid_aupr": aupr        # Optional
                    }
        
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_f1_max"], mode="max")
