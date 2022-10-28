# Adapted from Tevatron (https://github.com/texttron/tevatron)

import glob
import logging
import os
import random
from typing import Callable, Dict, List, Union

from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import BatchEncoding, PreTrainedTokenizer

from ..arguments import DataArguments, DRPretrainingDataArguments
from ..data_augmentation_strategy import (Cropping, NullStrategy,
                                          SequentialStrategies)
from ..trainer import DRTrainer

logger = logging.getLogger(__name__)


def build_one_data_pos_neg(
    example: Dict[str, Union[List[int], List[List[int]]]],
    encode_fn: Callable[[List[int], bool], BatchEncoding],
    hashed_seed: int,
    epoch: int,
    data_args: DataArguments
):
    qry = example['query']
    encoded_query = encode_fn(qry, is_query=True)
    encoded_passages = []
    group_positives = example['positives']
    group_negatives = example['negatives']

    if data_args.positive_passage_no_shuffle or hashed_seed is None:
        pos_psg = group_positives[0]
    else:
        pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
    encoded_passages.append(encode_fn(pos_psg))

    negative_size = data_args.train_n_passages - 1
    if len(group_negatives) < negative_size:
        if hashed_seed is not None:
            negs = random.choices(group_negatives, k=negative_size)
        else:
            negs = [x for x in group_negatives]
            negs = negs * 2
            negs = negs[:negative_size]
    elif data_args.train_n_passages == 1:
        negs = []
    elif data_args.negative_passage_no_shuffle:
        negs = group_negatives[:negative_size]
    else:
        _offset = epoch * negative_size % len(group_negatives)
        negs = [x for x in group_negatives]
        if hashed_seed is not None:
            random.Random(hashed_seed).shuffle(negs)
        negs = negs * 2
        negs = negs[_offset: _offset + negative_size]

    for neg_psg in negs:
        encoded_passages.append(encode_fn(neg_psg))

    assert len(encoded_passages) == data_args.train_n_passages

    return {"query": encoded_query, "passages": encoded_passages}


def build_one_data_pretrain(
    example: Dict[str, List[int]],
    encode_fn: Callable[[List[int], bool], BatchEncoding],
    data_args: DRPretrainingDataArguments
):
    content = example[data_args.pretrain_target_field]
    encoded_query = encode_fn(content, is_query=True)
    encoded_passages = [encode_fn(content)]

    return {"query": encoded_query, "passages": encoded_passages}


class TrainDatasetBase:
    '''
    Abstract base class for all train datasets in Openmatch.\n
    This implants arguments and data preparation, but should be mostly used for identifying an OpenMatch Train Dataset.\n
    All future dataset ABCs would subclass this and `(Iterable)Dataset`.
    '''

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        self._prepare_data(data_args, shuffle_seed, cache_dir)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.trainer = trainer

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = None
        self.dataset = None


class StreamTrainDataset(TrainDatasetBase, IterableDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        super(StreamTrainDataset, self).__init__(
            tokenizer, data_args, trainer, shuffle_seed, cache_dir)

    def __len__(self):
        concat_filenames = " ".join(self.data_files)
        count = 0
        with os.popen("wc -l {}".format(concat_filenames)) as f:
            for line in f:
                lc, filename = line.strip().split()
                lc = int(lc)
                if filename != "total":
                    count += lc
        return count

    def __iter__(self):
        raise NotImplementedError


class MappingTrainDataset(TrainDatasetBase, Dataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        super(MappingTrainDataset, self).__init__(
            tokenizer, data_args, trainer, shuffle_seed, cache_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raise NotImplementedError


class StreamDRTrainDataset(StreamTrainDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        super(StreamDRTrainDataset, self).__init__(
            tokenizer, data_args, trainer, shuffle_seed, cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(
            os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(
            seed=shuffle_seed, buffer_size=10_000) if shuffle_seed is not None else self.dataset

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            return build_one_data_pos_neg(example, self.create_one_example, hashed_seed, epoch, self.data_args)

        return process_fn

    def __iter__(self):
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(self.trainer.args.seed)
        self.dataset.set_epoch(epoch)
        return iter(self.dataset.map(self.get_process_fn(epoch, _hashed_seed), remove_columns=["positives", "negatives"]))


class StreamDRPretrainDataset(StreamTrainDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DRPretrainingDataArguments,
        trainer: DRTrainer = None,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        super(StreamDRPretrainDataset, self).__init__(
            tokenizer, data_args, trainer, shuffle_seed, cache_dir)
        pretrain_strategies_str = data_args.pretrain_strategies.split(
            ",") if data_args.pretrain_strategies is not None else []
        strategies = []
        for strategy_str in pretrain_strategies_str:
            if strategy_str == "null":
                strategies.append(NullStrategy())
                logger.info("Adding NullStrategy")
            elif strategy_str == "crop":
                strategies.append(Cropping(
                    ratio_min=data_args.cropping_ratio_min, ratio_max=data_args.cropping_ratio_max))
                logger.info("Adding Cropping, ratio_min={}, ratio_max={}".format(
                    data_args.cropping_ratio_min, data_args.cropping_ratio_max))
            else:
                raise ValueError(
                    "Unknown pretraining strategy: {}".format(strategy_str))
        self.apply_strategy = SequentialStrategies(*strategies)

        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(
            os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(
            seed=shuffle_seed, buffer_size=10_000) if shuffle_seed is not None else self.dataset

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        text_encoding = self.apply_strategy(text_encoding)
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            return build_one_data_pretrain(example, self.create_one_example, self.data_args)

        return process_fn

    def __iter__(self):
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(self.trainer.args.seed)
        self.dataset.set_epoch(epoch)
        return iter(self.dataset.map(self.get_process_fn(epoch, _hashed_seed), remove_columns=self.all_columns))


class MappingDRTrainDataset(MappingTrainDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        # No shuffle seed is needed for mapping datasets, but were keeped to maintain interface
        super(MappingDRTrainDataset, self).__init__(
            tokenizer, data_args, trainer, shuffle_seed, cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(
            os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __getitem__(self, item):
        group = self.dataset[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)
        return build_one_data_pos_neg(group, self.create_one_example, _hashed_seed, epoch, self.data_args)


class StreamDREvalDataset(StreamDRTrainDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        cache_dir: str = None
    ) -> None:
        super(StreamDREvalDataset, self).__init__(
            tokenizer, data_args, cache_dir=cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]

    def __iter__(self):
        return iter(self.dataset.map(self.get_process_fn(0, None), remove_columns=["positives", "negatives"]))


class StreamRRTrainDataset(StreamTrainDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        trainer: DRTrainer = None,
        shuffle_seed: int = None,
        cache_dir: str = None
    ) -> None:
        super(StreamRRTrainDataset, self).__init__(
            tokenizer, data_args, trainer, shuffle_seed, cache_dir)
        self.neg_num = 1

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(
            os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(
            seed=shuffle_seed, buffer_size=10_000) if shuffle_seed is not None else self.dataset

    def create_one_example(self, qry_encoding, psg_encoding) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            qry_encoding + psg_encoding,
            truncation='longest_first',
            max_length=self.data_args.q_max_len + self.data_args.p_max_len + 2,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            qry = example['query']
            group_positives = example['positives']
            group_negatives = example['negatives']

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(
                    hashed_seed + epoch) % len(group_positives)]
            encoded_pos_pair = self.create_one_example(qry, pos_psg)

            if hashed_seed is None:
                neg_psg = group_negatives[0]
            else:
                neg_psg = group_negatives[(
                    hashed_seed + epoch) % len(group_negatives)]
            encoded_neg_pair = self.create_one_example(qry, neg_psg)
            return {"pos_pair": encoded_pos_pair, "neg_pair": encoded_neg_pair}

        return process_fn

    def __iter__(self):
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(self.trainer.args.seed)
        self.dataset.set_epoch(epoch)
        return iter(self.dataset.map(self.get_process_fn(epoch, _hashed_seed), remove_columns=["positives", "negatives"]))


class StreamRREvalDataset(StreamRRTrainDataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        cache_dir: str = None
    ) -> None:
        super(StreamRREvalDataset, self).__init__(
            tokenizer, data_args, cache_dir=cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]

    def __iter__(self):
        return iter(self.dataset.map(self.get_process_fn(0, None), remove_columns=["positives", "negatives"]))
