import logging
import unittest

import gin
import numpy as np
import shutil as sh
import torch
from torchvision import transforms

logging.getLogger().setLevel(logging.INFO)

from meta_dataset.data.dataset_spec import DatasetSpecification
from meta_dataset.data.learning_spec import Split
from meta_dataset.datasets.datasets_test import make_dummy_dataset
from meta_dataset.datasets.datasets import make_multisource_episode_dataset


# DatasetSpecification to use in tests
def get_dataset_spec(path):
    DATASET_SPEC = DatasetSpecification(
        name=None,
        classes_per_split={
            Split.TRAIN: 15,
            Split.VALID: 5,
            Split.TEST: 10
        },
        images_per_class=dict(enumerate([10, 20, 30] * 10)),
        class_names=None,
        path=path,
        file_pattern='{}.h5')
    return DATASET_SPEC


# Define defaults and set Gin configuration for EpisodeDescriptionSampler
MIN_WAYS = 5
MAX_WAYS_UPPER_BOUND = 50
MAX_NUM_QUERY = 10
MAX_SUPPORT_SET_SIZE = 500
MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS = 100
MIN_LOG_WEIGHT = np.log(0.5)
MAX_LOG_WEIGHT = np.log(2)
DATASETS_WITH_EXAMPLE_SPLITS = ()
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
        self.split = Split.TRAIN
        self.num_episodes = 10
        self.spec1 = get_dataset_spec("tmp1")
        self.spec2 = get_dataset_spec("tmp2")
        make_dummy_dataset(self.spec1, 0)
        make_dummy_dataset(self.spec2, 42)
        self.epoch_size = 10

    def test_without_threading(self):
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(84),
                                        transforms.ToTensor()])

        dataset_specs = [self.spec1, self.spec2]
        use_dag_ontology_list = [False, False]
        use_bilevel_ontology_list = [False, False]
        split = Split.TRAIN
        image_size = 84
        dataset1 = make_multisource_episode_dataset(dataset_specs,
                                                    use_dag_ontology_list,
                                                    use_bilevel_ontology_list,
                                                    split,
                                                    self.epoch_size,
                                                    image_size,
                                                    transforms=transform)

        dataloader1 = torch.utils.data.DataLoader(dataset1, 1, num_workers=2, shuffle=False,
                                                  worker_init_fn=dataset1.setup)

        counter = 0
        dataset_ids = []
        for ep1 in dataloader1:
            examples1, labels1 = unpack_episode(ep1)
            dataset_ids.append(int((255 * examples1[0, :, 2, ...]).round().mean()))
            counter += 1
        self.assertEqual(counter, self.num_episodes)
        self.assertEqual(2, len(set(dataset_ids)))
        self.assertIn(0, set(dataset_ids))
        self.assertIn(42, set(dataset_ids))

    def tearDown(self):
        sh.rmtree("tmp1")
        sh.rmtree("tmp2")
