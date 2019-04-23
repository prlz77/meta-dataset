import cv2
import h5py
import numpy as np
import os
import shutil as sh
import unittest
import gin

from meta_dataset.datasets.datasets import HDF5ClassDataset
from meta_dataset.data.dataset_spec import DatasetSpecification
from meta_dataset.data.learning_spec import Split
from meta_dataset.data import sampling

from torchvision.transforms import transforms

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


def create_unique_image(id, clss, dataset_id):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :, 0] = id
    img[:, :, 1] = clss
    img[:, :, 2] = dataset_id
    return cv2.imencode(".png", img)[1][:, 0]


def make_dummy_dataset(dataset_spec, dataset_id=0):
    id = 0
    os.makedirs(dataset_spec.path, exist_ok=True)
    for clss, count in dataset_spec.images_per_class.items():
        filename = os.path.join(dataset_spec.path, dataset_spec.file_pattern.format(clss))
        fp = h5py.File(filename, 'w')
        dt = h5py.special_dtype(vlen=np.uint8)
        fp.create_dataset("images", dtype=dt, shape=(count,))
        fp.create_dataset("labels", dtype=np.uint32, shape=(count,))
        images = [create_unique_image(i + id, clss, dataset_id) for i in range(count)]
        fp['images'][...] = images
        fp["labels"][...] = [clss] * count
        fp.close()
        id += count


def unpack_episode(episode):
    examples = np.concatenate((episode["support_images"].numpy(), episode["query_images"].numpy()), 0) * 255
    targets = np.concatenate((episode["support_class_labels"].numpy(), episode["query_class_labels"].numpy()), 0)
    return examples, targets


class HDF5DatasetTest(unittest.TestCase):
    def setUp(self):
        os.makedirs("tmp", exist_ok=True)
        make_dummy_dataset(DATASET_SPEC)
        self.split = Split.TRAIN
        self.dataset_spec = DATASET_SPEC
        self.image_size = 8

    def check_episode_consistency(self, examples, targets):
        """Tests that a given episode is correctly built and consistent.

        In particular:
        - test that examples come from the right class

        Args:
          examples: A 1D array of strings.
          targets: A 1D array of ints.
        """
        self.check_consistent_class(examples, targets)
        self.assertEqual(len(examples), len(targets))

    def check_consistent_class(self, examples, targets):
        """Checks that the content of examples corresponds to the target.

        This assumes the datasets were generated from `construct_dummy_datasets`,
        with a dummy class of DUMMY_CLASS_ID with empty string examples.

        Args:
          examples: A 1D array of strings.
          targets: A 1D array of ints.
        """
        self.assertEqual(len(examples), len(targets))
        for (example, target) in zip(examples, targets):
            label = int(example[1, :, :].mean())
            self.assertEqual(label, target)

    def generate_and_check(self, sampler, num_episodes):
        episodes = self.generate_episodes(sampler, num_episodes)
        for episode in episodes:
            examples, targets = unpack_episode(episode)
            self.check_episode_consistency(examples, targets)

    def test_train(self):
        """Tests that a few episodes are consistent."""
        sampler = sampling.EpisodeDescriptionSampler(DATASET_SPEC, Split.TRAIN)
        self.generate_and_check(sampler, 10)

    def test_valid(self):
        sampler = sampling.EpisodeDescriptionSampler(DATASET_SPEC, Split.VALID)
        self.generate_and_check(sampler, 10)

    def test_test(self):
        sampler = sampling.EpisodeDescriptionSampler(DATASET_SPEC, Split.TEST)
        self.generate_and_check(sampler, 10)

    def test_fixed_query(self):
        sampler = sampling.EpisodeDescriptionSampler(
            DATASET_SPEC, self.split, num_query=5)
        self.generate_and_check(sampler, 10)

    def test_no_query(self):
        sampler = sampling.EpisodeDescriptionSampler(
            DATASET_SPEC, self.split, num_query=0)
        self.generate_and_check(sampler, 10)

    def test_fixed_shots(self):
        sampler = sampling.EpisodeDescriptionSampler(
            DATASET_SPEC, self.split, num_support=3, num_query=7)
        self.generate_and_check(sampler, 10)

    def test_fixed_ways(self):
        sampler = sampling.EpisodeDescriptionSampler(
            DATASET_SPEC, self.split, num_ways=12)
        self.generate_and_check(sampler, 10)

    def test_fixed_episodes(self):
        sampler = sampling.EpisodeDescriptionSampler(
            DATASET_SPEC, self.split, num_ways=12, num_support=3, num_query=7)
        self.generate_and_check(sampler, 10)

    def test_non_deterministic_shuffle(self):
        """Different Readers generate different episode compositions.

        Even with the same episode descriptions, the content should be different.
        """
        num_episodes = 10
        init_rng = sampling.RNG
        seed = 20181120
        episode_streams = []

        try:
            for _ in range(2):
                sampling.RNG = np.random.RandomState(seed)
                sampler = sampling.EpisodeDescriptionSampler(self.dataset_spec,
                                                             self.split)
                episodes = self.generate_episodes(sampler, num_episodes)
                episode_streams.append(episodes)
                for episode in episodes:
                    examples, targets = unpack_episode(episode)
                    self.check_episode_consistency(examples, targets)

        finally:
            # Restore the original RNG
            sampling.RNG = init_rng

        # It is unlikely that all episodes will be the same
        num_identical_episodes = 0
        for episode1, episode2 in zip(*episode_streams):
            examples1, targets1 = unpack_episode(episode1)
            examples2, targets2 = unpack_episode(episode2)
            self.check_episode_consistency(examples1, targets1)
            self.check_episode_consistency(examples2, targets2)
            np.testing.assert_array_equal(targets1, targets2)
            if np.equal(examples1, examples2).all():
                num_identical_episodes += 1

        self.assertNotEqual(num_identical_episodes, num_episodes)

    def test_deterministic_noshuffle(self):
        """Tests episode generation determinism when there is noshuffle queue."""
        num_episodes = 10
        init_rng = sampling.RNG
        seed = 20181120
        episode_streams = []
        try:
            for _ in range(2):
                sampling.RNG = np.random.RandomState(seed)
                sampler = sampling.EpisodeDescriptionSampler(self.dataset_spec,
                                                             self.split)
                episodes = self.generate_episodes(sampler, num_episodes, shuffle=False,
                                                  shuffle_seed=seed)
                episode_streams.append(episodes)
                for episode in episodes:
                    examples, targets = unpack_episode(episode)
                    self.check_episode_consistency(examples, targets)

        finally:
            # Restore the original RNG
            sampling.RNG = init_rng

        for episode1, episode2 in zip(*episode_streams):
            examples1, targets1 = unpack_episode(episode1)
            examples2, targets2 = unpack_episode(episode2)
            np.testing.assert_array_equal(examples1, examples2)
            np.testing.assert_array_equal(targets1, targets2)

    def test_deterministic_tfseed(self):
        """Tests episode generation determinism when shuffle queues are seeded."""
        num_episodes = 10
        seed = 20181120
        episode_streams = []
        chunk_sizes = []
        init_rng = sampling.RNG
        try:
            for _ in range(2):
                sampling.RNG = np.random.RandomState(seed)
                sampler = sampling.EpisodeDescriptionSampler(self.dataset_spec,
                                                             self.split)
                episodes = self.generate_episodes(
                    sampler, num_episodes, shuffle_seed=seed)
                episode_streams.append(episodes)
                chunk_size = sampler.compute_chunk_sizes()
                chunk_sizes.append(chunk_size)
                for episode in episodes:
                    examples, targets = unpack_episode(episode)
                    self.check_episode_consistency(examples, targets)

        finally:
            # Restore the original RNG
            sampling.RNG = init_rng

        self.assertEqual(chunk_sizes[0], chunk_sizes[1])

        for episode1, episode2 in zip(*episode_streams):
            examples1, targets1 = unpack_episode(episode1)
            examples2, targets2 = unpack_episode(episode2)
            self.check_episode_consistency(examples1, targets1)
            self.check_episode_consistency(examples2, targets2)
            np.testing.assert_array_equal(examples1, examples2)
            np.testing.assert_array_equal(targets1, targets2)

    def generate_episodes(self,
                          sampler,
                          num_episodes,
                          shuffle=True,
                          shuffle_seed=None):
        dataset_spec = sampler.dataset_spec
        split = sampler.split

        dataset = HDF5ClassDataset(dataset_spec, split, sampler=sampler, epoch_size=num_episodes, reshuffle=shuffle,
                                   image_size=self.image_size,
                                   transforms=transforms.ToTensor(),
                                   shuffle_seed=shuffle_seed)
        dataset.setup()
        dataset.build_episode_indices()

        episodes = [dataset[i] for i in range(num_episodes)]
        return episodes

    def check_description_episode_consistency(self, description, episode, offset):
        support_class_labels = episode["support_class_labels"]
        query_class_labels = episode["query_class_labels"]
        support_episode_labels = episode["support_episode_labels"]
        query_episode_labels = episode["query_episode_labels"]
        support_cursor = 0
        query_cursor = 0
        for i, (class_id, shot, query) in enumerate(description):
            start = support_cursor
            end = support_cursor + shot
            support_cursor = end
            s_cls = support_class_labels[start:end]
            s_epi = support_episode_labels[start:end]
            start = query_cursor
            end = query_cursor + query
            query_cursor = end
            q_cls = query_class_labels[start:end]
            q_epi = query_episode_labels[start:end]
            np.testing.assert_array_equal(s_cls, [class_id + offset] * shot)
            np.testing.assert_array_equal(q_cls, [class_id + offset] * query)
            np.testing.assert_array_equal(s_epi, [i] * shot)
            np.testing.assert_array_equal(q_epi, [i] * query)

    def check_same_as_generator(self, split, offset):
        """Tests that the targets are the one requested by the generator.

        Args:
          split: A value of the Split enum, which split to generate from.
          offset: An int, the difference between the absolute class IDs in the
            source, and the relative class IDs in the episodes.
        """
        num_episodes = 10
        seed = 20181121
        init_rng = sampling.RNG
        try:
            sampling.RNG = np.random.RandomState(seed)
            sampler = sampling.EpisodeDescriptionSampler(DATASET_SPEC, split)
            # Each description is a (class_id, num_support, num_query) tuple.
            descriptions = [
                sampler.sample_episode_description() for _ in range(num_episodes)
            ]

            sampling.RNG = np.random.RandomState(seed)
            sampler = sampling.EpisodeDescriptionSampler(DATASET_SPEC, split)
            episodes = self.generate_episodes(sampler, num_episodes)
            self.assertEqual(len(descriptions), len(episodes))
            for (description, episode) in zip(descriptions, episodes):
                self.check_description_episode_consistency(description, episode, offset)
        finally:
            sampling.RNG = init_rng

    def test_same_as_generator(self):
        # The offset corresponds to the difference between the absolute class ID as
        # used in the episode pipeline, and class ID relative to the split (provided
        # by the episode generator).
        offset = 0
        for split in Split:
            self.check_same_as_generator(split, offset)
            offset += len(DATASET_SPEC.get_classes(split))

    def tearDown(self):
        sh.rmtree(DATASET_SPEC.path)


if __name__ == "__main__":
    unittest.main()
