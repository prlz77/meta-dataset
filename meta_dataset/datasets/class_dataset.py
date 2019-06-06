import logging
import os
import resource
import time

import meta_dataset.data.sampling as sampling
import numpy as np
import torch
from meta_dataset.data.learning_spec import Split
from meta_dataset.datasets.episodic_dataloader import EpisodicDataLoader
from torch import multiprocessing
from torch.utils.data import Dataset
from tqdm import tqdm

resource.setrlimit(
  resource.RLIMIT_CORE,
  (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

# TODO (prlz77): make this configurable
torch.utils.data.DataLoader = EpisodicDataLoader
logging.warning("Extended dataloader __iter__ function for episodic training.")

obj = None
queue = None

def get_random_initializer():
  return np.frombuffer(os.urandom(4), dtype=np.uint32)[0]

# Helper functions to seed workers
def init_fn(base_seed, _queue, _obj, last):
  global obj
  global queue
  queue = _queue
  obj = _obj
  idx = queue.get(block=True)
  if idx + 1 != last:
    queue.put(idx + 1)
  seed = (idx + base_seed) % 2 ** 32
  sampling.RNG.seed(seed)
  obj.RNG = np.random.RandomState(seed=(seed))
  torch.manual_seed(seed)


def build_episode_indices(*args, **kwargs):
  global obj
  global queue
  ret = obj.build_episode_indices()
  queue.put(None, block=False)
  return ret


class ClassDataset(Dataset):
  """Specifies the methods to sample from individual classes in a dataset

  Meta Dataset requires sampling from individual classes, this class
  provides a common view for all the datasets that compose it. Corresponds to
  meta_dataset.data.reader

  Since the epoch is pre-computed with the function build_episode_indices,
  please, make sure to call it after each epoch if epochs are small to avoid
  repeating the same data. To automate this, use meta_dataset.datasets.episodic_dataloader.EpisodicDataLoader
  instead of torch.utils.data.DataLoader
  """

  def __init__(self, backend, dataset_spec, split, epoch_size, pool, reshuffle, shuffle_seed):
    """ Constructor
    Args:
        dataset_spec: meta_dataset.data.dataset_spec.DatasetSpecification instance
        split: meta_dataset.data.learning_spec.Split instance
        sampler: meta_dataset.data.sampling.Sampler instance
        epoch_size: the number of iterations of each epoch
        pool: meta_dataset pool option
        reshuffle: whether to reshuffle the images inside each class each iteration
        shuffle_seed: seed for the random generator. If fixed, examples will always
                      come in the same order given the same episode description
    """
    self.cache = None
    self.epoch_size = epoch_size
    self.name = dataset_spec.name
    self.class_set = list(dataset_spec.get_classes(split))
    self.num_classes = len(self.class_set)
    self.backend = backend

    """ 
    The dataset offset is modified by a Multisource Datataset
    so that labels become unique to each dataset
    """
    self.total_images_per_class = np.array([
      dataset_spec.get_total_images_per_class(self.class_set[class_idx], pool)
      for class_idx in range(self.num_classes)])

    self.cursors = np.zeros(self.num_classes, dtype=np.uint32)

    self.reshuffle = reshuffle

    if shuffle_seed is not None:
      self.RNG = np.random.RandomState(seed=shuffle_seed)
    else:
      self.RNG = np.random

    self._reshuffle_indices()

    if split == Split.TRAIN:
      self.offset = 0
      self.split = "train"
    elif split == Split.VALID:
      self.offset = dataset_spec.classes_per_split[Split.TRAIN]
      self.split = "valid"
    elif split == Split.TEST:
      self.offset = dataset_spec.classes_per_split[Split.TRAIN] + \
                    dataset_spec.classes_per_split[Split.VALID]
      self.split = "test"

    logging.info("Loaded %s split of %s: %d classes" % (split, self.name, self.num_classes))

  def setup(self, worker_id):
    self.backend.setup(worker_id)

  def build_episode_indices(self):
    """Pre-computes the indices and labels of the images to load during an
    epoch. Avoids using random seeds on the worker threads.
    """
    raise NotImplementedError

  def load_save_cache(self, cache_folder, epochs):
    """ Loads a cache with all the batch/episode indices or creates it if it does not exist

    Args:
      cache_folder: string. Directory where to save the cached indices (defaults to .cache)
      epochs: int. Generate indices for that many epochs

    Returns: list. The cache loaded in main memory

    """
    cache_folder = os.path.join(cache_folder, self.name)
    try:
      os.makedirs(cache_folder)
      self.save_cache(cache_folder, epochs)
    except OSError:
      self.load_cache(cache_folder, epochs)

  def load_cache(self, cache_folder, epochs):
    """ Loads a cache from the given folder

    Args:
      cache_folder: string. Path to the cache file
      epochs: int. Number of epochs that the cache should contain. Fails when less than required.

    Returns:

    """
    t = 0
    # TODO (pau): is there a better way to do this?
    while not os.path.exists(os.path.join(cache_folder, 'ready')):  # if another process is doing it, wait
      if t == 0:
        logging.info("Waiting for a process to finish the cache...")
      time.sleep(1)
      t += 1
      if t > 3600:
        raise TimeoutError
    self.cache = torch.load(os.path.join(cache_folder, "cache.pt"))
    assert (len(self.cache) >= epochs)
    logging.info("Loaded cache from %s" % cache_folder)

  def save_cache(self, cache_folder, epochs):
    """ Generates batch/episode indices and saves them into a torch file

    Args:
      cache_folder: string. folder where to save the cached indices
      epochs: int. number of epochs to generate

    Returns: list. Cached indices.

    """
    logging.info("Saving cache to %s" % cache_folder)
    queue = multiprocessing.Queue()
    queue.put(1)
    nworkers = 32
    last = nworkers + 1

    with multiprocessing.Pool(nworkers, initializer=init_fn, initargs=(get_random_initializer(), queue, self, last)) as pool:
      _cache = pool.map_async(build_episode_indices, range(epochs))
      for _ in tqdm(range(epochs)):
        queue.get(block=True)
      cache = _cache.get()
      del queue
    torch.save(cache, os.path.join(cache_folder, "cache.pt"))

    with open(os.path.join(cache_folder, 'ready'), 'w') as outfile:
      outfile.write('\n')

    self.cache = cache

  def _reshuffle_indices(self):
    """Helper procedure to to randomize the samples inside each class"""
    self.sample_indices = [self.RNG.permutation(self.total_images_per_class[i]) for i in
                           range(self.num_classes)]

  def read_class(self, class_id, indices):
    """Abstract method to load examples from disk given a class_id and
        the amount of samples

    Returns: a list of numpy arrays of size [amount, (height, width,
    channels)]

    Args:
        class_id: the class index
        indices: the indices of the samples inside the class
    """
    return self.backend.read_class(str(self.class_set[int(class_id)]), indices)

  def __len__(self):
    return self.epoch_size


class EpisodicClassDataset(ClassDataset):

  def __init__(self, backend, dataset_spec, split, sampler, epoch_size, pool, reshuffle, shuffle_seed):
    """ Constructor
    Args:
        dataset_spec: meta_dataset.data.dataset_spec.DatasetSpecification instance
        split: meta_dataset.data.learning_spec.Split instance
        sampler: meta_dataset.data.sampling.Sampler instance
        epoch_size: the number of iterations of each epoch
        pool: meta_dataset pool option
        reshuffle: whether to reshuffle the images inside each class each iteration
        shuffle_seed: seed for the random generator. If fixed, examples will always
                      come in the same order given the same episode description
    """
    super().__init__(backend, dataset_spec, split, epoch_size, pool, reshuffle, shuffle_seed)
    self.sampler = sampler
    self.episodic = True

  def build_episode_indices(self):
    """Pre-computes the indices and labels of the images to load during an
    epoch avoids using random seeds on the worker threads
    """
    if self.cache is not None:
      self.episodes = self.cache.pop(0)
      return self.episodes

    if self.reshuffle:
      self._reshuffle_indices()

    self.episodes = []

    # Adapted from meta_dataset.data.reader
    for _ in range(self.epoch_size):
      episode_description = self.sampler.sample_episode_description()
      episode = dict(
        class_idx=[],
        indices=[],
        support_class_labels=[],
        query_class_labels=[],
        support_episode_labels=[],
        query_episode_labels=[],
        shots=[],
        querys=[],
      )
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
        self.cursors[class_idx] = end
        indices = self.sample_indices[class_idx][start:end]
        support_class_labels = [class_idx + self.offset] * shots
        query_class_labels = [class_idx + self.offset] * query
        support_episode_labels = [i] * shots
        query_episode_labels = [i] * query
        total_support += shots
        total_query += query
        episode["class_idx"].append(class_idx)
        episode["indices"].append(indices)
        episode["support_class_labels"].extend(support_class_labels)
        episode["query_class_labels"].extend(query_class_labels)
        episode["support_episode_labels"].extend(support_episode_labels)
        episode["query_episode_labels"].extend(query_episode_labels)
        episode["shots"].append(shots)
        episode["querys"].append(query)

      for k in episode.keys():
        if k != "indices":
          episode[k] = np.array(episode[k])

      episode["name"] = self.name
      episode["ways"] = len(episode["shots"])
      self.episodes.append((total_support, total_query, self.name, episode))
    return self.episodes

  def __getitem__(self, item):
    """Reads an episode and returns it

    Returns: a dictionary with the episode

    Args:
        item: episode index in 0..(epoch_size - 1)
    """
    total_support, total_query, name, episode = self.episodes[item]
    self.episodes[item] = None  # Release memory

    for k in episode.keys():
      if k not in ["indices", "name", "ways"]:
        episode[k] = torch.from_numpy(episode[k])

    episode["support_images"] = []
    episode["query_images"] = []

    for i in range(len(episode["shots"])):
      shot = int(episode["shots"][i])
      im = self.read_class(episode["class_idx"][i], episode["indices"][i])
      episode["support_images"].extend(im[:shot])
      episode["query_images"].extend(im[shot:])

    episode["support_images"] = torch.stack(episode["support_images"], 0)
    episode["query_images"] = torch.stack(episode["query_images"], 0)
    return episode


class BatchClassDataset(ClassDataset):
  def __init__(self, backend, dataset_spec, split, num_train_classes, num_test_classes, epoch_size,
               batch_size, pool=None, reshuffle=True, shuffle_seed=None):
    super().__init__(backend, dataset_spec, split, epoch_size, pool, reshuffle, shuffle_seed)
    self.episodic = False
    self.batch_size = batch_size
    self.class_proportions = torch.from_numpy(self.total_images_per_class / self.total_images_per_class.sum())
    self.num_train_classes = num_train_classes
    self.num_test_classes = num_test_classes

  def build_episode_indices(self):
    """Pre-computes the indices and labels of the images to load during an
    epoch avoids using random seeds on the worker threads
    """
    if self.cache is not None:
      self.batches = self.cache.pop(0)
      return self.batches

    if self.reshuffle:
      self._reshuffle_indices()

    self.batches = []

    # Adapted from meta_dataset.data.reader
    for _ in range(self.epoch_size):
      batch = {}
      for i in range(self.batch_size):
        class_idx = int(torch.multinomial(self.class_proportions, 1))
        remaining = self.total_images_per_class[class_idx] - self.cursors[class_idx]
        if remaining == 0:
          self.cursors[class_idx] = 0
          if self.reshuffle:
            self.RNG.shuffle(self.sample_indices[class_idx])
        cursor = self.cursors[class_idx]
        self.cursors[class_idx] += 1

        img_index = self.sample_indices[class_idx][cursor]
        if class_idx in batch:
          batch[class_idx].append(img_index)
        else:
          batch[class_idx] = [img_index]
      for k in batch.keys():
        batch[k] = np.array(batch[k])
      self.batches.append(batch)
    return self.batches

  def __getitem__(self, item):
    """Reads an episode and returns it

    Returns: a tuple with the batch

    Args:
        item: int. Batch number.
    """
    batch = self.batches[item]

    images = []
    labels = []

    for class_id, indices in batch.items():
      images.extend(self.read_class(class_id, indices))
      labels.extend([class_id] * len(indices))

    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels, self.name
