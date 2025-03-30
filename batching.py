from collections import defaultdict
from typing import Optional
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
import numpy as np

MAX_LENGTH = 640


class BrainDataset(Dataset):
    """
    Batching with padding by MAX_LENGTH. Each batch has fixed length
    (batch_size x MAX_LENGTH).
    """
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = []
        with open(data_path, "r") as f:
            self.texts.extend(self.tokenizer(line.strip())["input_ids"] for line in f)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids = torch.tensor(self.texts[idx], dtype=torch.int64)
        t = torch.zeros((self.max_length), dtype=torch.int64)
        if self.max_length >= len(ids):
            t[:len(ids)] = ids
        else:
            t = ids[:self.max_length]
            t[-1] = ids[-1]
        return t


class BigBrainDataset(BrainDataset):
    """
    Same as the past, but with collate_fn. Now each batch has different size,
    because max length defined by text in current batch (so padding only in collate_fn).
    """
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__(data_path, max_length)

    def __getitem__(self, idx: int):
        return torch.tensor(self.texts[idx], dtype=torch.int64)


def collate_fn(
        batch: list[torch.Tensor], max_length: Optional[int] = MAX_LENGTH
) -> torch.Tensor:
    """
    Pad each sequence of the incoming sequences list.
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    length = max(batch, key=lambda x: x.shape[0]).shape[0]
    max_length = length if max_length > length else max_length
    result_batch = torch.zeros(len(batch), max_length, dtype=torch.int64)
    for i in range(result_batch.shape[0]):
        l = len(batch[i])
        if max_length >= l:
            result_batch[i, :l] = batch[i]
        else:
            l = max_length
            result_batch[i] = batch[i][:l]
            result_batch[i, -1] = batch[i][-1]

    return result_batch


class UltraBigBrainDataset(BigBrainDataset):
    """
    This way of batching has a batch_sampler. It group examples with of
    similar length into buckets, and sample examples for every batch
    from a single bucket.
    """
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__(data_path, max_length)


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, data, batch_size: int, k: int = 1, max_length: Optional[int] = MAX_LENGTH):
        self.len2bucket = {}
        self.buckets = defaultdict(list)
        self.lens = []
        for i in data:
            self.lens.append(len(i))
        min_ = len(min(data, key=lambda x: len(x)))
        max_ = len(max(data, key=lambda x: len(x)))
        max_ = max_length if max_ > max_length else max_
        self.bound = torch.arange(min_, max_ + 2 * k, k + 1)

        for i, ids in enumerate(data):
            l = len(ids)
            if l > max_:
                l = max_
            if l not in self.len2bucket:
                bucket_id = self.get_bucket(l)
                self.len2bucket[l] = bucket_id
            self.buckets[self.len2bucket[l]].append(i)

        self.batches = []
        for bucket_id in self.buckets:
            self.buckets[bucket_id] = torch.tensor(self.buckets[bucket_id])
            idx = torch.randperm(len(self.buckets[bucket_id]))
            self.buckets[bucket_id] = self.buckets[bucket_id][idx]
            split = self.buckets[bucket_id].split(batch_size)
            self.batches.extend(split)

        np.random.shuffle(self.batches)

    def get_bucket(self, length):
        mask = ~torch.logical_and(self.bound <= length, self.bound < length)
        return torch.min(torch.where(mask)[0]).item()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch.tolist()


class UltraDuperBigBrainDataset(IterableDataset):
    """
    Pack all sequences into one long sequence and generate metadata
    that indicates where each original sequence starts and ends.
    Similiar to FlashAttention approach.
    """
    def __init__(self, data_path: str, batch_size=32,
                max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.texts = []
        with open(data_path, "r") as f:
            self.texts.extend(self.tokenizer(line.strip())["input_ids"] for line in f)

        np.random.shuffle(self.texts)

        self.batches, self.mask_batches = [], []
        row, mask = [], []
        batch, mask_batch = [], []
        current_doc = 0
        for text in self.texts:
            res = self.max_length - len(row)
            if len(text) <= res:
                row.extend(text)
                mask.extend([current_doc] * len(text))
            else:
                row.extend(text[:res])
                row[-1] = text[-1]
                mask.extend([current_doc] * res)
            current_doc += 1

            if len(row) == self.max_length:
                batch.append(row)
                mask_batch.append(mask)
                current_doc = 0
                row, mask = [], []

            if len(batch) == self.batch_size:
                self.batches.append(torch.tensor(batch))
                self.mask_batches.append(torch.tensor(mask_batch))
                batch, mask_batch = [], []

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for i in range(len(self.batches)):
            yield self.batches[i], self.mask_batches[i]
