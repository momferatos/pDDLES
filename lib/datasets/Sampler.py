# Copyright (c) 2022-2026 Georgios Momferatos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math
import random
from lib.datasets.TurbDataset import TurbDataset

class TurbSampler(DistributedSampler):

    def __init__(self, dataset: TurbDataset, num_replicas: int | None = None,
                 rank: int | None = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # Per-rank sample count (stock DistributedSampler formula). The
        # dataset's __len__ is the true file count; it must not be
        # pre-divided by world_size or sampler-less DataLoaders shrink too.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible,
            # so each rank receives the same amount of data.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.length = len(self.dataset.filenames)
        self.shuffle = shuffle
        self.seed = seed
    

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.length, generator=g).tolist()
            # type: ignore[arg-type]
        else:
            indices = list(range(self.length))  # type: ignore[arg-type]

        if not self.drop_last:
            # pad up to total_size, not self.length: indices already has
            # self.length entries, so padding against it was a no-op and the
            # last rank got a short shard, desynchronizing DDP collectives.
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(
                    padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample: every rank gets exactly num_samples indices
        start = self.rank * self.num_samples
        end = (self.rank + 1) * self.num_samples
        indices = indices[start:end]
        self.dataset.load(indices)
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        # The per-rank shard (self.indices) is fixed at construction time, since
        # only those files are loaded into memory by dataset.load(). To still get
        # per-epoch variation, reshuffle the *order* of this rank's shard each
        # epoch using seed + epoch (set via set_epoch()).
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.indices), generator=g).tolist()
            indices = [self.indices[i] for i in perm]
        else:
            indices = self.indices

        return iter(indices)
    
        
    # If len stays the same you can leave it out, else you can also modify it
    def __len__(self) -> int:
        return self.num_samples
