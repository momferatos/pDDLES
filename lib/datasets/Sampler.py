import torch
from torch.utils.data.distributed import DistributedSampler
import math
import random
from lib.datasets.TurbDataset import TurbDataset

class TurbSampler(DistributedSampler):
    
    def __init__(self, dataset, num_replicas=None,
                 rank=None, shuffle=True,
                 seed=0, drop_last=False):
        
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
        self.num_samples = len(self.dataset)

        
        # #If the dataset length is evenly divisible by # of
        # # replicas, then there
        # # is no need to drop any data, since the dataset
        # # will be split equally.
        # if self.drop_last and len(self.dataset) % self.num_replicas != 0:
        #     # type: ignore[arg-type]
        #     # Split to nearest available length that is evenly divisible.
        #     # This is to ensure each rank receives
        #     # the same amount of data when
        #     # using this Sampler.
        #     self.num_samples = math.ceil(
        #         (len(self.dataset) - self.num_replicas) / self.num_replicas
        #         # type: ignore[arg-type]
        #     )
        # else:
        #     self.num_samples = math.ceil(
        #         len(self.dataset) / self.num_replicas)
        #     # type: ignore[arg-type]

        
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
            # add extra samples to make it evenly divisible
            padding_size = self.length - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(
                    padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.length]
        assert len(indices) == self.length

        # subsample
        start = self.rank * self.num_samples
        end = min((self.rank + 1) * self.num_samples, self.length)
        indices = indices[start:end]
        self.dataset.load(indices)
        self.indices = indices

    def __iter__(self): 
        # Here you need to look how DistributedSampler implements __iter__
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/
        # distributed.html#DistributedSampler
        # Then you can modify it to serve your purposes
    
        return iter(self.indices)
    
        
    # If len stays the same you can leave it out, else you can also modify it
    def __len__(self):
        return self.num_samples
