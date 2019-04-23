import unittest

import gin
import numpy as np
import os
import shutil as sh
import torch
from copy import deepcopy
from torchvision import transforms
import time
import logging

logging.getLogger().setLevel(logging.INFO)

from meta_dataset.data import sampling
from meta_dataset.data.dataset_spec import DatasetSpecification
from meta_dataset.data.learning_spec import Split
from meta_dataset.datasets.datasets import HDF5ClassDataset
from meta_dataset.datasets.datasets_test import make_dummy_dataset

# DatasetSpecification to use in tests
DATASET_SPEC = DatasetSpecification(
    name=None,
    classes_per_split={
        Split.TRAIN: 15,
        Split.VALID: 5,
        Split.TEST: 10
    },
    images_per_class=dict(enumerate([10, 20, 30] * 10)),
    class_names=None,
    path="tmp",
    file_pattern='{}.h5')

# Define defaults and set Gin configuration for EpisodeDescriptionSampler
MIN_WAYS = 5
MAX_WAYS_UPPER_BOUND = 50
MAX_NUM_QUERY = 10
MAX_SUPPORT_SET_SIZE = 500
MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS = 100
MIN_LOG_WEIGHT = np.log(0.5)
MAX_LOG_WEIGHT = np.log(2)
gin.bind_parameter('EpisodeDescriptionSampler.min_ways', MIN_WAYS)
gin.bind_parameter('EpisodeDescriptionSampler.max_ways_upper_bound',
                   MAX_WAYS_UPPER_BOUND)
gin.bind_parameter('EpisodeDescriptionSampler.max_num_query', MAX_NUM_QUERY)
gin.bind_parameter('EpisodeDescriptionSampler.max_support_set_size',
                   MAX_SUPPORT_SET_SIZE)
gin.bind_parameter(
    'EpisodeDescriptionSampler.max_support_size_contrib_per_class',
    MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS)
gin.bind_parameter('EpisodeDescriptionSampler.min_log_weight', MIN_LOG_WEIGHT)
gin.bind_parameter('EpisodeDescriptionSampler.max_log_weight', MAX_LOG_WEIGHT)


def unpack_episode(episode):
    """ Helper function that extracts samples and labels from an episode in numpy
    Args:
        episode: the episode to unpack
    """
    examples = torch.cat([episode["support_images"], episode["query_images"]], 1).numpy()
    labels = torch.cat([episode["support_class_labels"], episode["query_class_labels"]], 1).numpy()
    return examples, labels


class EpisodicDataLoaderTest(unittest.TestCase):
    def setUp(self):
        os.makedirs("tmp", exist_ok=True)
        make_dummy_dataset()
        self.split = Split.TRAIN
        self.dataset_spec = DATASET_SPEC
        self.num_episodes = 10

    def test_without_threading(self):

        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(84),
                                        transforms.ToTensor()])
        sampling.RNG.seed(1234)
        sampler = sampling.EpisodeDescriptionSampler(self.dataset_spec,
                                                     self.split)
        dataset1 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes,
                                    image_size=84,
                                    transforms=transform,
                                    shuffle_seed=1234)
        read1 = []

        dataset1.setup()
        dataloader1 = torch.utils.data.DataLoader(dataset1, 1, num_workers=0, shuffle=False)

        counter = 0
        for ep1 in dataloader1:
            examples1, labels1 = unpack_episode(ep1)
            read1.append((examples1, labels1))
            counter += 1
        self.assertEqual(counter, self.num_episodes)

        sampling.RNG.seed(1234)
        dataset2 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes,
                                    image_size=84,
                                    transforms=transform,
                                    shuffle_seed=1234)
        dataloader2 = torch.utils.data.DataLoader(dataset2, 1, num_workers=0, shuffle=False)
        dataset2.setup()
        read2 = []
        counter = 0
        for ep2 in dataloader2:
            examples1, labels1 = unpack_episode(ep2)
            read2.append((examples1, labels1))
            counter += 1
        self.assertEqual(counter, self.num_episodes)

        for ((ex1, l1), (ex2, l2)) in zip(read1, read2):
            np.testing.assert_array_equal(ex1, ex2)
            np.testing.assert_array_equal(l1, l2)

    def test_with_threading(self):

        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(84),
                                        transforms.ToTensor()])
        sampling.RNG.seed(1234)
        sampler = sampling.EpisodeDescriptionSampler(self.dataset_spec,
                                                     self.split)
        dataset1 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes,
                                    image_size=84,
                                    transforms=transform,
                                    shuffle_seed=1234)
        read1 = []

        dataloader1 = torch.utils.data.DataLoader(dataset1, 1, num_workers=2, shuffle=False,
                                                  worker_init_fn=dataset1.setup)

        counter = 0
        for ep1 in dataloader1:
            examples1, labels1 = unpack_episode(ep1)
            read1.append((examples1, labels1))
            counter += 1
        self.assertEqual(counter, self.num_episodes)

        sampling.RNG.seed(1234)
        dataset2 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes,
                                    image_size=84,
                                    transforms=transform,
                                    shuffle_seed=1234)
        dataloader2 = torch.utils.data.DataLoader(dataset2, 1, num_workers=2, shuffle=False,
                                                  worker_init_fn=dataset2.setup)
        read2 = []
        counter = 0
        for ep2 in dataloader2:
            examples1, labels1 = unpack_episode(ep2)
            read2.append((examples1, labels1))
            counter += 1
        self.assertEqual(counter, self.num_episodes)

        for ((ex1, l1), (ex2, l2)) in zip(read1, read2):
            np.testing.assert_array_equal(ex1, ex2)
            np.testing.assert_array_equal(l1, l2)

    def test_shuffled_without_threading(self):

        sampler = sampling.EpisodeDescriptionSampler(self.dataset_spec,
                                                     self.split)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(84),
                                        transforms.ToTensor()])
        dataset1 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes,
                                    image_size=84,
                                    transforms=transform,
                                    shuffle_seed=1234)
        dataset2 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes,
                                    image_size=84,
                                    transforms=transform,
                                    shuffle_seed=1234)
        dataset1.setup()
        dataset2.setup()
        dataloader1 = torch.utils.data.DataLoader(dataset1, 1, num_workers=0, shuffle=True)
        dataloader2 = torch.utils.data.DataLoader(dataset2, 1, num_workers=0, shuffle=True)

        counter = 0
        different = 0
        for ep1, ep2 in zip(dataloader1, dataloader2):
            examples1, labels1 = unpack_episode(ep1)
            examples2, labels2 = unpack_episode(ep2)
            try:
                np.testing.assert_array_equal(examples1, examples2)
            except AssertionError:
                different += 1
            counter += 1

        self.assertEqual(counter, self.num_episodes)
        self.assertNotEqual(different, 0)

    def test_epoch_prefetch(self):
        sampler = sampling.EpisodeDescriptionSampler(self.dataset_spec,
                                                     self.split)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(84),
                                        transforms.ToTensor()])
        dataset1 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes,
                                    image_size=84,
                                    transforms=transform,
                                    reshuffle=False,
                                    shuffle_seed=1234)
        dataset1.setup()
        dataset1.build_episode_indices()
        dataloader1 = torch.utils.data.DataLoader(dataset1, 1, num_workers=0, shuffle=True)

        indices = deepcopy(dataset1.episode_indices)
        for i in range(self.num_episodes):
            np.testing.assert_array_equal(indices[i][2][0][1], dataset1.episode_indices[i][2][0][1])
        for _ in dataloader1:
            break
        for i in range(self.num_episodes):
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_array_equal,
                                     indices[i][2][0][1],
                                     dataset1.episode_indices[i][2][0][1])

    def test_shuffled_with_threading(self):
        sampler = sampling.EpisodeDescriptionSampler(self.dataset_spec,
                                                     self.split)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(84),
                                        transforms.ToTensor()])
        dataset1 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes,
                                    image_size=84,
                                    transforms=transform,
                                    shuffle_seed=1234)
        dataset2 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes,
                                    image_size=84,
                                    transforms=transform,
                                    shuffle_seed=1234)

        dataloader1 = torch.utils.data.DataLoader(dataset1, 1, num_workers=2, shuffle=True,
                                                  worker_init_fn=dataset1.setup)
        dataloader2 = torch.utils.data.DataLoader(dataset2, 1, num_workers=2, shuffle=True,
                                                  worker_init_fn=dataset2.setup)

        counter = 0
        different = 0
        for ep1, ep2 in zip(dataloader1, dataloader2):
            examples1, labels1 = unpack_episode(ep1)
            examples2, labels2 = unpack_episode(ep2)
            try:
                np.testing.assert_array_equal(examples1, examples2)
            except AssertionError:
                different += 1
            counter += 1

        self.assertEqual(counter, self.num_episodes)
        self.assertNotEqual(different, 0)

    def test_threading_faster(self):
        sampler = sampling.EpisodeDescriptionSampler(self.dataset_spec,
                                                     self.split)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(84),
                                        transforms.ToTensor()])
        dataset1 = HDF5ClassDataset(self.dataset_spec, self.split,
                                    sampler=sampler, epoch_size=self.num_episodes * 10,
                                    image_size=84,
                                    transforms=transform,
                                    shuffle_seed=1234)

        dataloader1 = torch.utils.data.DataLoader(dataset1, 1, num_workers=2, shuffle=True,
                                                  worker_init_fn=dataset1.setup)

        threaded = 0
        t = time.time()
        for _ in dataloader1:
            threaded += time.time() - t
            t = time.time()

        logging.info("Threaded time %.03fs" % threaded)
        dataset1.setup()
        dataloader2 = torch.utils.data.DataLoader(dataset1, 1, num_workers=0, shuffle=True, )
        nothreaded = 0
        t = time.time()
        for _ in dataloader2:
            nothreaded += time.time() - t
            t = time.time()

        logging.info("Sequential time %.03fs" % nothreaded)
        self.assertGreater(nothreaded, threaded)

    def tearDown(self):
        sh.rmtree(DATASET_SPEC.path)
