import torch
from torch.utils.data import Dataset
import gin


@gin.configurable('BatchSplitReaderGetReader', whitelist=['add_dataset_offset'])
class MultisourceEpisodeDataset(Dataset):
    def __init__(self, datasets, epoch_size, add_dataset_offset=False):
        """ Creates a dataset from multiple Dataset instances

        Each episode is sampled from a randomly chosen dataset.
        All the datasets will prefetch the same number of episode indices,
        which is the total iteration length of the MultiSourceEpisodeDataset
        to avoid running out of examples from the individual datasets.

        Args:
            datasets: a list of pytorch datasets
            epoch_size: the number of iterations per epoch
        """
        self.datasets = datasets
        self.epoch_size = epoch_size

        offset = 0
        for dataset in datasets:
            dataset.epoch_size = self.epoch_size
            if add_dataset_offset:
              dataset.offset = offset
              offset += dataset.num_classes

    def build_episode_indices(self):
        """ Generates the indices for all the episodes in an epoch.

        It does it by forwarding the call to each individual dataset.
        """
        for dataset in self.datasets:
            dataset.build_episode_indices()

    def setup(self, worker_id=0):
        """ Thread initialization function.

        Operations performed here will be executed individually in each thread.

        Args:
            worker_id: the thread unique identifier
        """
        for dataset in self.datasets:
            dataset.setup(worker_id)

    def __getitem__(self, item):
        """ Sample an episode from a randomly chosen dataset

        Args:
            item: episode number inside the epoch

        Returns: dict(arrays) a fully-assembled episode

        """
        dataset_idx = int(torch.randint(len(self.datasets), (1,)))
        dataset = self.datasets[dataset_idx]
        return dataset[item]

    def __len__(self):
        """ tells the iterator the amount of iterations per epoch

        Returns: int, the amount of iterations per epoch

        """
        return self.epoch_size
