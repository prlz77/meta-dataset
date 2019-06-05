import logging
import os

import h5py
import meta_dataset.data as data
import meta_dataset.data.sampling as sampling
from meta_dataset.datasets.multisource_datasets import MultisourceEpisodeDataset
from meta_dataset.datasets.backends import Hdf5Backend
from meta_dataset.datasets.class_dataset import EpisodicClassDataset, BatchClassDataset
import meta_dataset.data.learning_spec as learning_spec
import torchvision.transforms as transforms_lib
import numpy as np
import cv2
import torch
from multiprocessing.pool import ThreadPool


def imdecode(im):
  return cv2.imdecode(im, cv2.IMREAD_COLOR)


def make_one_source_batch_dataset(dataset_spec,
                                  split,
                                  num_train_classes,
                                  num_test_classes,
                                  epoch_size,
                                  batch_size,
                                  pool=None,
                                  reshuffle=True,
                                  image_size=None,
                                  transforms=None):
  """Returns a pipeline emitting data from one single source as Batches.
  Args:
    dataset_spec: A DatasetSpecification object defining what to read from.
    split: A learning_spec.Split object identifying the source split.
    batch_size: An int representing the max number of examples in each batch.
    pool: String (optional), for example-split datasets, which example split to
      use ('valid', or 'test'), used at meta-test time only.
    reshuffle: bool. Whether to reshuffle indices before sampling the images.
    image_size: int, desired image size used during decoding.
  Returns:
    A Dataset instance that outputs decoded batches from all classes in the
    split.
  """

  if ".h5" in dataset_spec.file_pattern:
    backend = Hdf5Backend(dataset_spec, image_size, transforms)

  dataset = BatchClassDataset(backend, dataset_spec, split, num_train_classes,
                              num_test_classes,
                              epoch_size, batch_size, pool,
                              reshuffle=reshuffle,
                              shuffle_seed=None)

  return dataset

def make_multisource_batch_pipeline(dataset_spec_list,
                                    split,
                                    num_train_classes,
                                    num_test_classes,
                                    epoch_size,
                                    batch_size,
                                    add_dataset_offset,
                                    pool=None,
                                    reshuffle=True,
                                    image_size=None,
                                    transforms=None):
  """Returns a pipeline emitting data from multiple source as Batches.

  Args:
    dataset_spec_list: A list of DatasetSpecification, one for each source.
    split: A learning_spec.Split object identifying the source split.
    batch_size: An int representing the max number of examples in each batch.
    add_dataset_offset: A Boolean, whether to add an offset to each dataset's
      targets, so that each target is unique across all datasets.
    pool: String (optional), for example-split datasets, which example split to
      use ('valid', or 'test'), used at meta-test time only.
    shuffle_buffer_size: int or None, number of examples in the buffer used for
      shuffling the examples from different classes, while they are mixed
      together. There is only one shuffling operation, not one per class.
    read_buffer_size_bytes: int or None, buffer size for each TFRecordDataset.
    image_size: int, desired image size used during decoding.

  Returns:
    A Dataset instance that outputs decoded batches from all classes in the
    split.
  """
  sources = []
  for dataset_spec in dataset_spec_list:
    if ".h5" in dataset_spec.file_pattern:
      backend = Hdf5Backend(dataset_spec, image_size, transforms)

    dataset = BatchClassDataset(backend, dataset_spec, split, num_train_classes,
                                num_test_classes,
                                epoch_size, batch_size, pool,
                                reshuffle=reshuffle,
                                shuffle_seed=None)
    sources.append(dataset)

  return MultisourceEpisodeDataset(sources, epoch_size=epoch_size, add_dataset_offset=add_dataset_offset)

def make_one_source_episode_pipeline(dataset_spec,
                                     use_dag_ontology,
                                     use_bilevel_ontology,
                                     split,
                                     epoch_size,
                                     image_size,
                                     pool=None,
                                     num_ways=None,
                                     num_support=None,
                                     num_query=None,
                                     reshuffle=True,
                                     transforms=None):
  """Returns a pipeline emitting data from one single source as Episodes.

  Args:
    dataset_spec: A DatasetSpecification object defining what to read from.
    use_dag_ontology: Whether to use source's ontology in the form of a DAG to
      sample episodes classes.
    use_bilevel_ontology: Whether to use source's bilevel ontology (consisting
      of superclasses and subclasses) to sample episode classes.
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
  else:
    use_all_classes = False

  sampler = sampling.EpisodeDescriptionSampler(
    dataset_spec,
    split,
    pool=pool,
    use_dag_hierarchy=use_dag_ontology,
    use_bilevel_hierarchy=use_bilevel_ontology,
    use_all_classes=use_all_classes,
    num_ways=num_ways,
    num_support=num_support,
    num_query=num_query)

  if ".h5" in dataset_spec.file_pattern:
    backend = Hdf5Backend(dataset_spec, image_size, transforms)
  dataset = EpisodicClassDataset(backend, dataset_spec, split, sampler,
                                 epoch_size, pool,
                                 reshuffle=reshuffle,
                                 shuffle_seed=None)
  return dataset


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
      backend = Hdf5Backend(dataset_spec, image_size, transforms[dataset_spec.name])

    dataset = EpisodicClassDataset(backend, dataset_spec, split, sampler,
                                   epoch_size, pool,
                                   reshuffle=reshuffle,
                                   shuffle_seed=None)
    sources.append(dataset)

  return MultisourceEpisodeDataset(sources, epoch_size=epoch_size)
