# Adapted from Tevatron (https://github.com/texttron/tevatron)

import copy
import importlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (AutoConfig, AutoModel, BatchEncoding,
                          PreTrainedModel, T5EncoderModel)
from transformers.modeling_outputs import ModelOutput

from ..arguments import DataArguments
from ..arguments import DRTrainingArguments as TrainingArguments
from ..arguments import ModelArguments
from .custom_models import T5ModelWithFusion
from .nano_t5 import MyT5RoPE
from .rope_t5 import T5ModelRoPE
from ..utils import mean_pooling
from .linear import LinearHead


logger = logging.getLogger(__name__)


@dataclass
class DROutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class DRModel(nn.Module):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            tied: bool = True,
            maxp: int = None,
            fusion: int = None,
            feature: str = "last_hidden_state",
            pooling: str = "first",
            head_q: nn.Module = None,
            head_p: nn.Module = None,
            normalize: bool = False,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.tied = tied
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.head_q = head_q
        self.head_p = head_p

        self.feature = feature
        self.pooling = pooling
        self.normalize = normalize

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.maxp = maxp
        self.fusion = fusion

        if train_args is not None:
            if train_args.distillation:
                self.loss_fn = nn.MSELoss() if train_args.distil_mode == "pairwise" else nn.KLDivLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

            if train_args.negatives_x_device:
                if not dist.is_initialized():
                    raise ValueError(
                        'Distributed training has not been initialized for representation all gather.')
                self.process_rank = dist.get_rank()
                self.world_size = dist.get_world_size()

    def _get_config_dict(self):
        config = {
            "tied": self.tied,
            "plm_backbone": {
                "type": type(self.lm_q).__name__,
                "feature": self.feature,
            },
            "pooling": self.pooling,
            "linear_head": bool(self.head_q),
            "normalize": self.normalize,
        }
        return config

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
            positive: Dict[str, Tensor] = None,
            negative: Dict[str, Tensor] = None,
            score: Tensor = None,
    ):

        q_hidden, q_reps = self.encode_query(
            query)  # (batch_size, hidden_size)

        if self.train_args.distillation:

            if self.train_args.distil_mode == "pairwise":

                pos_hidden, pos_reps = self.encode_passage(positive, self.fusion)
                neg_hidden, neg_reps = self.encode_passage(negative, self.fusion)
                scores_pos = torch.sum(q_reps * pos_reps, dim=1)
                scores_neg = torch.sum(q_reps * neg_reps, dim=1)
                margin_pred = scores_pos - scores_neg
                loss = self.loss_fn(margin_pred, score)
                return DROutput(q_reps=q_reps, p_reps=pos_reps, loss=loss, scores=torch.stack([scores_pos, scores_neg], dim=1))

            else:  # listwise
                # (batch_size * n_passages, hidden_size)
                p_hidden, p_reps = self.encode_passage(passage, self.fusion)
                batch_size = q_reps.shape[0]
                # (batch_size, n_passages, hidden_size)
                p_reps = p_reps.view(batch_size, -1, p_reps.shape[-1])
                # (batch_size, n_passages, hidden_size)
                q_reps_expanded = q_reps.unsqueeze(
                    1).expand(-1, p_reps.shape[1], -1)
                # (batch_size, n_passages)
                scores_pred = torch.sum(q_reps_expanded * p_reps, dim=2)
                scores_pred = F.log_softmax(scores_pred, dim=1)
                score = F.softmax(score, dim=1)
                loss = self.loss_fn(scores_pred, score)
                return DROutput(q_reps=q_reps, p_reps=p_reps, loss=loss, scores=scores_pred)

        else:

            p_hidden, p_reps = self.encode_passage(passage, self.fusion)

            if q_reps is None or p_reps is None:
                return DROutput(
                    q_reps=q_reps,
                    p_reps=p_reps
                )

            # if self.training:
            if self.train_args.negatives_x_device:
                q_reps = self.dist_gather_tensor(q_reps)
                p_reps = self.dist_gather_tensor(p_reps)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device \
                else self.train_args.per_device_train_batch_size

            scores = torch.matmul(q_reps, p_reps.transpose(0, 1))

            if self.maxp:
                scores = torch.max(scores.view(scores.size(0),int(scores.size(1)/self.maxp), self.maxp), dim=2).values

            target = torch.arange(
                scores.size(0),
                device=scores.device,
                dtype=torch.long
            )
            target = target * self.data_args.train_n_passages

            loss = self.loss_fn(scores, target)

            if self.training and self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            return DROutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps
            )

    def encode(self, items, model, head, is_q=False, fusion=None):
        if items is None:
            return None, None
        items = BatchEncoding(items)
        # Use decoder, if not configured to use encoder only AND if model is not an encoder-only model
        if ("T5" in type(model).__name__) and (not self.model_args.encoder_only) and (type(model).__name__ != "T5EncoderModel"):
            decoder_input_ids = torch.zeros(
                (items.input_ids.shape[0], 1), dtype=torch.long).to(items.input_ids.device)
            
            if fusion:

                items_out = model(
                    **items, decoder_input_ids=decoder_input_ids, fusion=fusion, return_dict=True)
            else:
                items_out = model(
                    **items, decoder_input_ids=decoder_input_ids, return_dict=True)
            if hasattr(items_out,'last_hidden_state'):
                hidden = items_out.last_hidden_state
                reps = hidden[:, 0, :]
            else:
                #hidden = items_out.decoder_hidden_states[-1]
                hidden = items_out.decoder_hidden_states
                reps = hidden[:, 0, :]
        elif "CLIP" in type(model).__name__:
            reps = hidden = items_out = model.get_text_features(
                **items, return_dict=True) if is_q else model.get_image_features(**items, return_dict=True)
        else:
            items_out = model(**items, return_dict=True)
            hidden = getattr(items_out, self.feature)
            if self.pooling == "first":
                reps = hidden[:, 0, :]
            elif self.pooling == "mean":
                reps = mean_pooling(hidden, items.attention_mask)
            elif self.pooling == "no":
                reps = hidden
            else:
                raise ValueError(
                    "Unknown pooling type: {}".format(self.pooling))
        if head is not None:
            reps = head(reps)  # D * d
        if self.normalize:
            reps = F.normalize(reps, dim=1)
        return hidden, reps

    def encode_passage(self, psg, fusion):
        return self.encode(psg, self.lm_p, self.head_p, fusion=fusion)

    def encode_query(self, qry):
        return self.encode(qry, self.lm_q, self.head_q)

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            model_name_or_path: str = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        model_name_or_path = model_name_or_path or model_args.model_name_or_path
        # load local
        config = None
        head_q = head_p = None
        if os.path.exists(os.path.join(model_name_or_path, "openmatch_config.json")):
            with open(os.path.join(model_name_or_path, "openmatch_config.json")) as f:
                config = json.load(f)

        if os.path.isdir(model_name_or_path) and config is not None:  # an OpenMatch model
            tied = config["tied"]
            if tied:
                logger.info(f'loading model weight from {model_name_or_path}')
                model_name = config["plm_backbone"]["type"]

                try:
                    model_class = getattr(
                        importlib.import_module("transformers"), model_name)
                except AttributeError:
                    try:
                        model_class = getattr(
                            importlib.import_module(".custom_models", package=__package__), model_name)
                    except AttributeError:
                        try:
                            model_class = getattr(
                                importlib.import_module(".nano_t5", package=__package__), model_name)
                        except AttributeError:
                            model_class = getattr(
                                importlib.import_module(".rope_t5", package=__package__), model_name)


                lm_q = lm_p = model_class.from_pretrained(
                    model_name_or_path,
                    **hf_kwargs
                )
                if config["linear_head"]:
                    head_q = head_p = LinearHead.load(model_name_or_path)
            else:
                _qry_model_path = os.path.join(
                    model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(
                    model_name_or_path, 'passage_model')
                _qry_head_path = os.path.join(model_name_or_path, 'query_head')
                _psg_head_path = os.path.join(
                    model_name_or_path, 'passage_head')

                logger.info(
                    f'loading query model weight from {_qry_model_path}')
                model_name = config["plm_backbone"]["lm_q_type"]
                model_class = getattr(
                    importlib.import_module("transformers"), model_name)
                if os.path.exists(os.path.join(_qry_model_path, "config.json")):
                    logger.info(
                        f'loading query model config from {_qry_model_path}')
                    qry_model_config = AutoConfig.from_pretrained(
                        _qry_model_path)
                    hf_kwargs["config"] = qry_model_config
                lm_q = model_class.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )

                logger.info(
                    f'loading passage model weight from {_psg_model_path}')
                model_name = config["plm_backbone"]["lm_p_type"]
                model_class = getattr(
                    importlib.import_module("transformers"), model_name)
                if os.path.exists(os.path.join(_psg_model_path, "config.json")):
                    logger.info(
                        f'loading passage model config from {_psg_model_path}')
                    psg_model_config = AutoConfig.from_pretrained(
                        _psg_model_path)
                    hf_kwargs["config"] = psg_model_config
                lm_p = model_class.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )

                if config["linear_head"]:
                    head_q = LinearHead.load(_qry_head_path)
                    head_p = LinearHead.load(_psg_head_path)
        else:  # a Huggingface model
            tied = not model_args.untie_encoder

            if model_args.encoder_only:
                model_class = T5EncoderModel
            elif model_args.fusion:
                model_class = T5ModelWithFusion
                print("Loading FiD T5")
            elif model_args.rope:
                model_class = T5ModelRoPE
                print("Loading RoPE T5!!!!")
            else:
                model_class = AutoModel
            try:
                lm_q = model_class.from_pretrained(model_name_or_path, **hf_kwargs)
                print("Loaded!!! Class:")
                print(model_class)
            except AttributeError as e:
                print("Custom implementation without from_pretrained. Manually loading from config and state dict.")
                lm_q = model_class(AutoConfig.from_pretrained(model_name_or_path))
                lm_q.load_state_dict(torch.load(f"{model_name_or_path}/pytorch_model.bin"), strict=False)
                print("done")
            lm_p = copy.deepcopy(lm_q) if not tied else lm_q
            if model_args.add_linear_head:
                head_q = LinearHead(
                    model_args.projection_in_dim, model_args.projection_out_dim)
                head_p = copy.deepcopy(head_q) if not tied else head_q

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            tied=tied,
            feature=model_args.feature if config is None else config["plm_backbone"]["feature"],
            pooling=model_args.pooling if config is None else config["pooling"],
            head_q=head_q,
            head_p=head_p,
            normalize=model_args.normalize if config is None else config["normalize"],
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
            maxp=model_args.maxp,
            fusion=model_args.fusion
        )
        return model

    def save(self, output_dir: str):
        if not self.tied:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(
                os.path.join(output_dir, 'passage_model'))
            if self.head_q is not None:
                self.head_q.save(os.path.join(output_dir, 'query_head'))
                self.head_p.save(os.path.join(output_dir, 'passage_head'))
        else:
            self.lm_q.save_pretrained(output_dir)
            if self.head_q is not None:
                self.head_q.save(output_dir)

        with open(os.path.join(output_dir, 'openmatch_config.json'), 'w') as f:
            json.dump(self._get_config_dict(), f, indent=4)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DRModelForInference(DRModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.eval()

    @torch.no_grad()
    def encode_passage(self, psg, fusion):
        return super(DRModelForInference, self).encode_passage(psg, fusion)

    @torch.no_grad()
    def encode_query(self, qry):
        return super(DRModelForInference, self).encode_query(qry)

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage, self.fusion)
        return DROutput(q_reps=q_reps, p_reps=p_reps)
