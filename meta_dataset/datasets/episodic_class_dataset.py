import numpy as np
import torch
import logging
from torch.utils.data import Dataset

from meta_dataset.datasets.episodic_dataloader import EpisodicDataLoader
from meta_dataset.data.learning_spec import Split

# TODO (prlz77): make this configurable
torch.utils.data.DataLoader = EpisodicDataLoader
logging.warning("Extended dataloader __iter__ function for episodic training.")


class EpisodicClassDataset(Dataset):
    """Specifies the methods to sample from individual classes in a dataset

    Meta Dataset requires sampling from individual classes, this class
    provides a common view for all the datasets that compose it. Corresponds to
    meta_dataset.data.reader

    Since the epoch is pre-computed with the function build_episode_indices,
    please, make sure to call it after each epoch if epochs are small to avoid
    repeating the same data. To automate this, use meta_dataset.datasets.episodic_dataloader.EpisodicDataLoader
    instead of torch.utils.data.DataLoader
    """

    def __init__(self, dataset_spec, split, sampler, image_size, epoch_size, pool, reshuffle, shuffle_seed):
        """ Constructor
        Args:
            dataset_spec: meta_dataset.data.dataset_spec.DatasetSpecification instance
            split: meta_dataset.data.learning_spec.Split instance
            sampler: meta_dataset.data.sampling.Sampler instance
            image_size: the output image size
            epoch_size: the number of iterations of each epoch
            pool: meta_dataset pool option
            reshuffle: whether to reshuffle the images inside each class each iteration
            shuffle_seed: seed for the random generator. If fixed, examples will always
                          come in the same order given the same episode description
        """
        self.sampler = sampler
        self.epoch_size = epoch_size
        self.image_size = image_size

        self.class_set = dataset_spec.get_classes(split)
        self.num_classes = len(self.class_set)

        self.total_images_per_class = dict(
            (class_idx,
             dataset_spec.get_total_images_per_class(self.class_set[class_idx], pool))
            for class_idx in range(self.num_classes))

        self.cursors = np.zeros(self.num_classes, dtype=np.uint32)

        self.episode_indices = []
        self.reshuffle = reshuffle

        if shuffle_seed is not None:
            self.RNG = np.random.RandomState(seed=shuffle_seed)
        else:
            self.RNG = np.random

        self._reshuffle_indices()

        if split == Split.TRAIN:
            self.offset = 0
        elif split == Split.VALID:
            self.offset = dataset_spec.classes_per_split[Split.TRAIN]
        elif split == Split.TEST:
            self.offset = dataset_spec.classes_per_split[Split.TRAIN] + \
                          dataset_spec.classes_per_split[Split.VALID]

    def _reshuffle_indices(self):
        """Helper procedure to to randomize the samples inside each class"""
        self.sample_indices = [self.RNG.permutation(self.total_images_per_class[i]) for i in
                               range(self.num_classes)]

    def build_episode_indices(self):
        """Pre-computes the indices and labels of the images to load during an
        epoch avoids using random seeds on the worker threads
        """
        if self.reshuffle:
            self._reshuffle_indices()

        self.episode_indices = []

        # Adapted from meta_dataset.data.reader
        for _ in range(self.epoch_size):
            episode_description = self.sampler.sample_episode_description()
            episode_indices = []
            total_support = 0
            total_query = 0
            for i, (class_idx, shots, query) in enumerate(episode_description):
                if shots + query > self.total_images_per_class[class_idx]:
                    raise ValueError("Requesting more images than what's available for the "
                                     'whole class')
                    # If the number of requested examples is greater than the number of
                    # examples remaining for the current pass over class `class_idx`, we flush
                    # the remaining examples and start a new pass over class `class_idx`.
                    # TODO(lamblinp): factor this out into its own tracker class for
                    # readability and testability.
                requested = shots + query
                remaining = self.total_images_per_class[class_idx] - self.cursors[class_idx]
                if requested > remaining:
                    self.cursors[class_idx] = 0
                if self.reshuffle:
                    self.RNG.shuffle(self.sample_indices[class_idx])
                start = self.cursors[class_idx]
                end = self.cursors[class_idx] + requested
                indices = self.sample_indices[class_idx][start:end]
                support_class_labels = np.array([class_idx + self.offset] * shots)
                query_class_labels = np.array([class_idx + self.offset] * query)
                support_episode_labels = np.array([i] * shots)
                query_episode_labels = np.array([i] * query)
                total_support += shots
                total_query += query
                episode_indices.append((class_idx, indices,
                                        support_class_labels,
                                        query_class_labels,
                                        support_episode_labels,
                                        query_episode_labels,
                                        shots, query))
                self.cursors[class_idx] += shots + query
            self.episode_indices.append((total_support, total_query, episode_indices))

    def read_class(self, class_id, indices):
        """Abstract method to load examples from disk given a class_id and
            the amount of samples

        Returns: a list of numpy arrays of size [amount, (height, width,
        channels)]

        Args:
            class_id: the class index
            indices: the indices of the samples inside the class
        """
        raise NotImplementedError

    def __getitem__(self, item):
        """Reads an episode and returns it

        Returns: a dictionary with the episode

        Args:
            item: episode index in 0..(epoch_size - 1)
        """
        total_support, total_query, episode = self.episode_indices[item]

        ret = dict(support_images=[],
                   query_images=[],
                   support_class_labels=[],
                   support_episode_labels=[],
                   query_class_labels=[],
                   query_episode_labels=[],
                   shots=[],
                   querys=[])

        for (class_idx, indices, support_class_labels, query_class_labels,
             support_episode_labels, query_episode_labels, shots, query) in episode:
            images = self.read_class(class_idx, indices)
            ret["support_images"].append(images[:shots])
            ret["query_images"].append(images[shots:])
            ret["support_class_labels"].append(torch.from_numpy(support_class_labels))
            ret["query_class_labels"].append(torch.from_numpy(query_class_labels))
            ret["support_episode_labels"].append(torch.from_numpy(support_episode_labels))
            ret["query_episode_labels"].append(torch.from_numpy(query_episode_labels))
            ret["shots"].append(torch.LongTensor([shots]))
            ret["querys"].append(torch.LongTensor([query]))

        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)

        ret["ways"] = len(episode)

        return ret

    def __len__(self):
        return self.epoch_size
