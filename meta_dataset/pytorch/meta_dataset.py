import gin
import sys
from os.path import dirname, abspath
import torch
from torchvision.transforms import transforms

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
from meta_dataset.utils.argparse import argparse
from meta_dataset.datasets.utils import get_benchmark_specification
import meta_dataset.datasets.datasets as datasets_lib
import meta_dataset.data.config
from meta_dataset.data import learning_spec
import meta_dataset.learner
import os
from functools import partial
import logging
import hashlib
import shutil
import cv2
import numpy as np

FLAGS = argparse.FLAGS


def get_split_enum(split):
  """Returns the Enum value corresponding to the given split.

  Args:
    split: A String.

  Raises:
    ValueError: Split must be one of 'train', 'valid' or 'test'.
  """
  # Get the int representing the chosen split.
  if split == 'train':
    split_enum = learning_spec.Split.TRAIN
  elif split == 'valid':
    split_enum = learning_spec.Split.VALID
  elif split == 'test':
    split_enum = learning_spec.Split.TEST
  else:
    raise ValueError('Split must be one of "train", "valid" or "test".')
  return split_enum


@gin.configurable
class LearnConfig(object):
  """
  From the original Metadataset, used to load gin parameters
  """

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


@gin.configurable
class LearnerConfig(object):
  """
  From the original Metadataset, used to load gin parameters
  """

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


@gin.configurable('benchmark')
def get_datasets(datasets=''):
  """Gets the list of dataset names.

  Args:
    datasets: A string of comma separated dataset names.

  Returns:
    A list of dataset names.
  """
  return [d.strip() for d in datasets.split(',')]


def parse_augmentation(augmentation_spec, image_size):
  """ Loads the data augmentation configuration

  Args:
      augmentation_spec: DataAugmentation instance
      image_size: the output image size

  Returns: torchvision.transforms object with the corresponding augmentation

  """

  def gaussian_noise(x, std):
    """ Helper function to add gaussian noise

    Args:
        x: input
        std: standard deviation for the normal distribution

    Returns: perturbed image

    """
    x += torch.randn(x.size()) * std
    return x

  def rescale(x):
    """ Rescales the image between -1 and 1

    Args:
        x: image

    Returns: rescaled image

    """
    return (x * 2) - 1

  _transforms = []
  if augmentation_spec.enable_gaussian_noise and \
      augmentation_spec.gaussian_noise_std > 0:
    f = partial(gaussian_noise, std=augmentation_spec.gaussian_noise_std)
    _transforms.append(transforms.Lambda(f))
  if augmentation_spec.enable_jitter and \
      augmentation_spec.jitter_amount > 0:
    _transforms.append(transforms.ToPILImage())
    amount = augmentation_spec.jitter_amount
    _transforms.append(transforms.RandomCrop(image_size,
                                             padding=amount))
  return _transforms


@gin.configurable('process_episode')
def get_transforms(name,
                   image_size,
                   support_data_augmentation=None,
                   query_data_augmentation=None,
                   *args,
                   **kwargs):
  """ Uses gin to produce the corresponding torchvision transforms

  Args:
      image_size: input image size
      support_data_augmentation: support set DataAugmentation specification
      query_data_augmentation: query set DataAugmentation specification
      *args: consume the rest of gin arguments
      **kwargs: consume the rest of gin arguments

  Returns:

  """
  # Numpy transforms
  support_transforms = []
  query_transforms = []
  if name in ["quickdraw", "omniglot"]:
    size = int(np.ceil(image_size / 32.)) * 32 + 1
    support_transforms.append(transforms.Lambda(lambda im: cv2.resize(im, (size, size), cv2.INTER_CUBIC)))
    query_transforms.append(transforms.Lambda(lambda im: cv2.resize(im, (size, size), cv2.INTER_CUBIC)))
  support_transforms += parse_augmentation(support_data_augmentation, image_size)
  query_transforms += parse_augmentation(query_data_augmentation, image_size)
  # PIL transforms
  support_transforms.append(transforms.ToTensor())
  # Tensor transforms
  return support_transforms, query_transforms


@gin.configurable('process_batch', whitelist=['batch_data_augmentation'])
def process_batch(*args, **kwargs):
  """ To consume unnecessary gin config

  Args:
      *args:
      **kwargs:

  Returns:

  """
  pass


@gin.configurable('Trainer')
class MetaDataset(object):
  """ Wrapper for MetaDataset

  Triantafillou, Eleni, et al. "Meta-dataset: A dataset of datasets for
  learning to learn from few examples." arXiv preprint arXiv:1903.03096 (2019).

  Similar to meta_dataset.trainer.Trainer but without the tensorflow boilerplate.
  It is copied here to avoid importing the rest of modules and to make gin work properly.
  """

  def __init__(self, **kwargs):
    """ Constructor, loads configuration from gin

    Args:
        **kwargs: gin arguments
    """
    for k, v in kwargs.items():
      setattr(self, k, v)

    self.learner_config = LearnerConfig()
    self.datasets = get_datasets()
    self.train_benchmark_spec, self.valid_benchmark_spec = \
      get_benchmark_specification(self.datasets,
                                  FLAGS.records_root_dir,
                                  FLAGS.eval_imbalance_dataset,
                                  self.data_config.image_height, )

    self.support_transforms = {}
    self.query_transforms = {}
    for dataset in self.datasets:
      support_transforms, query_transforms = get_transforms(dataset, self.data_config.image_height)
      self.support_transforms[dataset] = support_transforms
      self.query_transforms[dataset] = support_transforms

    if len(self.query_transforms) > 0:
      logging.warning("Different transforms for the query set not supported. We fallback to same transform.")

    self.support_transforms = {k: transforms.Compose(v) for k,v in self.support_transforms.items()}

    if self.valid_benchmark_spec is None:
      # This means that ImageNet is not a dataset in the given benchmark spec.
      # In this case the validation will be carried out on randomly-sampled
      # episodes from the meta-validation sets of all given datasets.
      self.valid_benchmark_spec = self.train_benchmark_spec

    self.eval_split = 'test'
    if self.is_training:
      self.required_splits = ['train', 'valid']
    else:
      self.required_splits = [self.eval_split]

    # Get the training, validation and testing specifications.
    # Each is either an EpisodeSpecification or a BatchSpecification.
    split_episode_or_batch_specs = {}
    if 'train' in self.required_splits:
      split_episode_or_batch_specs['train'] = self._create_train_specification()
    for split in ['valid', 'test']:
      if split not in self.required_splits:
        continue
      split_episode_or_batch_specs[split] = self._create_held_out_specification(
        split)
    self.split_episode_or_batch_specs = split_episode_or_batch_specs

  def _create_train_specification(self):
    """Returns an EpisodeSpecification or BatchSpecification for training."""
    if self.learner_config.episodic:
      return learning_spec.EpisodeSpecification(
        learning_spec.Split.TRAIN, self.num_train_classes,
        self.num_train_examples, self.num_test_examples)
    else:
      return learning_spec.BatchSpecification(learning_spec.Split.TRAIN,
                                              self.learn_config.batch_size)

  def _create_held_out_specification(self, split='test'):
    """Create an EpisodeSpecification for either validation or testing.

    Note that testing is done episodically whether or not training was episodic.
    This is why the different subclasses should not override this method.

    Args:
      split: one of 'valid' or 'test'

    Returns:
      an EpisodeSpecification.

    Raises:
      ValueError: Invalid split.
    """
    split_enum = get_split_enum(split)
    return learning_spec.EpisodeSpecification(split_enum, self.num_test_classes,
                                              self.num_train_examples,
                                              self.num_test_examples)

  def build_dataset(self, split):
    if self.learner_config.episodic:
      return self.build_episodic_dataset(split)
    else:
      return self.build_batch_dataset(split)

  def build_batch_dataset(self, split):
    if split in ["valid", "test"]:
      return self.build_episodic_dataset(split)

    # batches are only used during training
    benchmark_spec = self.train_benchmark_spec
    _, image_shape, dataset_spec_list, has_dag_ontology, has_bilevel_ontology = benchmark_spec
    dataset_split, batch_size = self.split_episode_or_batch_specs[split]
    epoch_size = self.learn_config.validate_every if split == "train" else self.learn_config.num_eval_episodes

    total_classes = 0
    for dataset_spec in dataset_spec_list:
      for _split in learning_spec.Split:
        total_classes += len(dataset_spec.get_classes(_split))

    if len(dataset_spec_list) == 1:
      dataset = datasets_lib.make_one_source_batch_dataset(
        dataset_spec_list[0],
        split=dataset_split,
        num_train_classes=total_classes,
        num_test_classes=self.num_test_classes,
        epoch_size=epoch_size,
        batch_size=batch_size,
        reshuffle=self.data_config.shuffle_buffer_size > 0,
        image_size=image_shape,
        transforms=self.support_transforms[dataset_spec_list[0].name])
    elif len(dataset_spec_list) > 1:
      dataset = datasets_lib.make_multisource_batch_dataset(
        dataset_spec_list,
        split=dataset_split,
        num_train_classes=total_classes,
        num_test_classes=self.num_test_classes,
        epoch_size=epoch_size,
        batch_size=batch_size,
        reshuffle=self.data_config.shuffle_buffer_size > 0,
        image_size=image_shape,
        transforms=self.support_transforms)
    else:
      raise ValueError("Empty list of datasets")

    self.maybe_save_cache(dataset, split)

    return dataset

  def maybe_save_cache(self, dataset, split):
    if FLAGS.use_cached_episodes:
      m = hashlib.md5()
      sources = []
      for root, _, files in os.walk("meta_dataset_pytorch/meta_dataset/learn"):
        for file in sorted(files):
          filename = os.path.join(root, file)
          if ".gin" == os.path.splitext(filename)[1]:
            sources.append(filename)
      for root, _, files in os.walk("meta_dataset_pytorch/meta_dataset/datasets"):
        for file in sorted(files):
          filename = os.path.join(root, file)
          if ".py" == os.path.splitext(filename)[1] and "episodic" in os.path.splitext(filename)[0]:
            sources.append(filename)
      for root, _, files in os.walk("meta_dataset_pytorch/meta_dataset/pytorch"):
        for file in sorted(files):
          filename = os.path.join(root, file)
          if ".py" == os.path.splitext(filename)[1]:
            sources.append(filename)
      for source in sorted(sources):
        with open(source, 'rb') as infile:
          m.update(infile.read())
      critical_flags = []
      for flag in critical_flags:
        m.update(str(getattr(FLAGS, flag)).encode())

      if FLAGS.cache_dir is not None:
        cache_root = FLAGS.cache_dir
      else:
        cache_root = '.cache'
      dirname = os.path.join(cache_root, 'metadataset', m.hexdigest(),
                             "%s_%s" % (split, "episodic" if dataset.episodic else "batched"))
      if FLAGS.force_cache and os.path.isdir(dirname):
        logging.info("Cache rebuild forced")
        shutil.rmtree(dirname)
      if os.path.isdir(FLAGS.reuse_cache):
        shutil.move(FLAGS.reuse_cache, dirname)
        logging.info("The code in episodic dataloader changed but user specified to use previous cache")

      dataset.load_save_cache(dirname, FLAGS.epochs)

  def build_episodic_dataset(self, split):
    """ Constructs an Episodic dataset with a single or multiple sources

    Args:
        split: meta_dataset.data.learning_spec.Split

    Returns: torch.utils.data.Dataset instance

    """
    benchmark_spec = (self.valid_benchmark_spec if split == 'valid' else self.train_benchmark_spec)
    _, image_shape, dataset_spec_list, has_dag_ontology, has_bilevel_ontology = benchmark_spec

    episode_spec = self.split_episode_or_batch_specs[split]
    dataset_split, num_classes, num_train_examples, num_test_examples = \
      episode_spec

    epoch_size = self.learn_config.validate_every if split == "train" else self.learn_config.num_eval_episodes

    use_dag_ontology = has_dag_ontology[0]
    if len(dataset_spec_list) == 1:
      dataset = datasets_lib.make_one_source_episode_dataset(
        dataset_spec_list[0],
        use_dag_ontology=use_dag_ontology,
        use_bilevel_ontology=has_bilevel_ontology[0],
        split=dataset_split,
        epoch_size=epoch_size,
        image_size=image_shape,
        num_ways=num_classes,
        num_support=num_train_examples,
        num_query=num_test_examples,
        reshuffle=self.data_config.shuffle_buffer_size > 0,
        transforms=self.support_transforms[dataset_spec_list[0].name])
    elif len(dataset_spec_list) > 1:
      dataset = datasets_lib.make_multisource_episode_dataset(
        dataset_spec_list,
        use_dag_ontology=use_dag_ontology,
        use_bilevel_ontology=has_bilevel_ontology,
        split=dataset_split,
        epoch_size=epoch_size,
        image_size=image_shape,
        num_ways=num_classes,
        num_support=num_train_examples,
        num_query=num_test_examples,
        reshuffle=self.data_config.shuffle_buffer_size > 0,
        transforms=self.support_transforms)
    else:
      raise ValueError("Empty list of datasets")

    self.maybe_save_cache(dataset, split)
    return dataset


def episodic_main():
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  meta_dataset = MetaDataset(is_training=FLAGS.is_training)
  test_dataset = meta_dataset.build_dataset("test")
  test_dataset.build_episode_indices()
  from copy import deepcopy
  episodes = deepcopy(test_dataset.episodes)
  meta_dataset = MetaDataset(is_training=True)
  train_dataset = meta_dataset.build_dataset("train")
  train_dataset.build_episode_indices()
  episodes_train = deepcopy(train_dataset.episodes)
  val_dataset = meta_dataset.build_dataset('valid')
  val_dataset.build_episode_indices()
  episodes_val = deepcopy(val_dataset.episodes)

  # valid_dataset = meta_dataset.build_dataset("valid")
  # valid_dataset.setup()
  # valid_dataset.build_episode_indices()
  # episode = valid_dataset[0]
  pass


def batch_main():
  gin_config = ['meta_dataset_pytorch/meta_dataset/learn/gin/best/baseline_imagenet.gin']
  gin.parse_config_files_and_bindings(gin_config, FLAGS.gin_bindings)
  meta_dataset = MetaDataset(is_training=FLAGS.is_training)
  train_dataset = meta_dataset.build_dataset("train")
  train_dataset.setup()
  train_dataset.build_episode_indices()
  data = train_dataset[0]
  pass


if __name__ == "__main__":
  parser = argparse.parser
  parser.add_argument('--gin_config', nargs='+',
                      default=['meta_dataset_pytorch/meta_dataset/learn/gin/default/prototypical_imagenet.gin'])
  parser.add_argument('--gin_bindings', nargs='+', default=[],
                      help="Commandline overrides for the gin configuration")
  parser.add_argument('--eval_imbalance_dataset', action="store_true",
                      help='A dataset on which to perform evaluation for assessing \
                              how class imbalance affects performance in binary episodes. \
                              By default it is empty and no imbalance analysis is performed.')
  parser.add_argument('--test_only', action='store_false', dest="is_training",
                      help="Whether we are training or only testing")
  logging.getLogger().setLevel(logging.INFO)
  parser.parse_args()
  batch_main()
