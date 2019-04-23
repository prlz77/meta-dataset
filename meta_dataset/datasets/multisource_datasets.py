import torch
from torch.utils.data import Dataset

import meta_dataset.data as data
import meta_dataset.data.sampling as sampling
from meta_dataset.datasets.datasets import HDF5ClassDataset


def make_multisource_episode_dataset(dataset_spec_list,
                                     use_dag_ontology_list,
                                     use_bilevel_ontology_list,
                                     split,
                                     epoch_size,
                                     image_size,
                                     pool=None,
                                     num_ways=None,
                                     num_support=None,
                                     num_query=None,
                                     reshuffle=True,
                                     transforms=None):
    """Adapted from the original metadataset tensorflow code Returns a pipeline
    emitting data from multiple sources as Episodes.

    Each episode only contains data from one single source. For each episode,
    its source is sampled uniformly across all sources.

    Args:
        dataset_spec_list: A list of DatasetSpecification, one for each source.
        use_dag_ontology_list: A list of Booleans, one for each source: whether
            to use that source's DAG-structured ontology to sample episode
            classes.
        use_bilevel_ontology_list: A list of Booleans, one for each source:
            whether to use that source's bi-level ontology to sample episode
            classes.
        split: A learning_spec.Split object identifying the sources split. It is
            the same for all datasets.
        epoch_size: The amount of loop iterations.
        image_size: The output image size
        pool: String (optional), for example-split datasets, which example split
            to use ('train', 'valid', or 'test'), used at meta-test time only.
        num_ways: Integer (optional), fixes the number of classes ("ways") to be
            used in each episode if provided.
        num_support: Integer (optional), fixes the number of examples for each
            class in the support set if provided.
        num_query: Integer (optional), fixes the number of examples for each
            class in the query set if provided.
        reshuffle: bool, whether to shuffle the images inside each class.
        transforms: List of functions, pre-processing functions to apply to the
            images inside each class.

    Returns:
        A Dataset instance that outputs fully-assembled and decoded episodes.
    """
    if pool is not None:
        if not data.POOL_SUPPORTED:
            raise NotImplementedError('Example-level splits or pools not supported.')
    sources = []
    for (dataset_spec, use_dag_ontology, use_bilevel_ontology) in zip(
            dataset_spec_list, use_dag_ontology_list, use_bilevel_ontology_list):
        sampler = sampling.EpisodeDescriptionSampler(
            dataset_spec,
            split,
            pool=pool,
            use_dag_hierarchy=use_dag_ontology,
            use_bilevel_hierarchy=use_bilevel_ontology,
            num_ways=num_ways,
            num_support=num_support,
            num_query=num_query)

        if ".h5" in dataset_spec.file_pattern:
            dataset = HDF5ClassDataset(dataset_spec, split, sampler, image_size,
                                       epoch_size, pool,
                                       reshuffle=reshuffle,
                                       transforms=transforms)
        sources.append(dataset)

    return MultisourceEpisodeDataset(sources, epoch_size=epoch_size)


class MultisourceEpisodeDataset(Dataset):
    def __init__(self, datasets, epoch_size):
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

        for dataset in datasets:
            dataset.epoch_size = self.epoch_size

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
