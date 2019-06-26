import h5py
import numpy as np
import os
import torchvision.transforms as transforms_lib
import logging
import cv2
from torch.multiprocessing import Queue, Process
import queue
import logging
import gin


def imdecode(im):
  return cv2.imdecode(im, cv2.IMREAD_COLOR)

@gin.configurable()
def Backend(*args, type="random_access", **kwargs):
  if type == "hdf5_random_access":
    return RandomAccessHdf5Backend(*args, **kwargs)
  elif type == "hdf5_sequential_access":
    return SequentialAccessHdf5Backend(*args, **kwargs)

class BaseBackend(object):
  def setup(self):
    raise NotImplementedError

  def read_class(self, class_id, indices):
    raise NotImplementedError


class RandomAccessHdf5Backend(BaseBackend):
  """Defines a dataset as a series of h5 files grouped by class in the same
  folder
  """

  def __init__(self, dataset_spec, split, image_size, transforms=None, fix_missing_images=True):
    """Initializes the hdf5 backend

    Args:
        dataset_spec: an instance from meta_dataset.data.dataset_spec
            describing the input dataset
        image_size: the output image size
        transforms: a function that applies successive transforms to the
            image
        fix_missing_images: the dataset converter sometimes fails to
            read all the images, so the real number of images per class
            is slightly different from the theoretical one.
    """

    self.path = os.path.join(dataset_spec.path, "{}.h5".format(dataset_spec.name))
    self.image_size = image_size
    self.transforms = transforms_lib.Compose([transforms_lib.Lambda(imdecode), transforms])
    self.split = split
    logging.warning(" Ignoring missing images")
    # self.check_missing_images(fix_missing_images)

  def check_missing_images(self, fix_missing_images):
    """ Checks if the number of examples per class in the dataset spec is the same
        as in the hdf5. If not, it corrects the theoretical number.

    Args:
        fix_missing_images: whether to fail or to fix image counts

    """
    errors = 0
    for i, name in enumerate(self.class_set):
      with h5py.File(self.path, 'r') as h5fp:
        if h5fp[str(name)].shape[0] != self.total_images_per_class[i]:
          self.total_images_per_class[i] = h5fp[str(name)].shape[0]
          errors += 1
        if errors > 1 and not fix_missing_images:
          raise RuntimeError("The number of stored images differs with the count in the DatasetSpec")
    total = int(100 * errors / self.num_classes)
    if total > 0:
      logging.warning("".join([" {}% of the classes didn't match the theoretical",
                               " number of examples per class in dataset {}."]).format(total, self.name))

  def setup(self, worker_id=None):
    """ Thread init function, file pointers are initialized here
        to avoid sharing them between threads

        This should be called manually when not using threading.

    Args:
        worker_id: unique identifier for the thread

    """
    #self.h5fp = h5py.File(self.path, 'r')

  def postprocess(self, x):
    """Helper function to ensure that the episode is returned in the correct
    order (samples, channels, h, w)

    Returns: postprocessed episode

    Args:
        x: the episode
    """
    return x

  def read_class(self, class_id, indices):
    """Reads the indexed images from a given class

    Returns: a numpy array with the shape len(indices)xcxhxw

    Args:
        class_id: the class from which to read
        indices: the indices of the images to load
    """

    unique_indices = np.unique(indices)
    sorted_indices = np.sort(unique_indices).tolist()
    with h5py.File(self.path, 'r') as h5fp:
      dataset = h5fp[class_id]
      images = dataset[sorted_indices, ...]
    images = [self.transforms(im) for im in images]

    if len(unique_indices) < len(indices):
      buffer = [None] * len(indices)
      for i, j in enumerate(indices):
        buffer[i] = images[sorted_indices.index(j)]
      return buffer
    else:
      return images

  def __del__(self):
    if hasattr(self, "h5fp"):
      try:
        self.h5fp.close()
      except:
        pass


class MasterHdf5Reader(Process):
  def __init__(self, dataset_spec, classes, nworkers, buffer_size=1000):
    super().__init__(daemon=True)
    self.worker_queues = [Queue() for _ in range(nworkers)]
    self.request_queue = Queue()
    self.path = os.path.join(dataset_spec.path, "{}.h5".format(dataset_spec.name))
    self.classes = list(map(str, classes))
    self.buffer_size = buffer_size

  def get_worker_queues(self):
    return self.worker_queues

  def get_request_queue(self):
    return self.request_queue

  def setup(self):
    self.cursors = {k: 0 for k in self.classes}
    self.buffers = {k: [] for k in self.classes}
    with h5py.File(self.path, 'r') as h5fp:
      self.buffer_sizes = {k: min(self.buffer_size, len(h5fp[k])) for k in self.classes}
    self.to_fill = self.classes[:]

  def fill_buffer(self, class_id):
    max_buffer_size = self.buffer_sizes[class_id]
    current_buffer_length = len(self.buffers[class_id])
    cursor = self.cursors[class_id]
    if current_buffer_length < max_buffer_size:
      h5fp = h5py.File(self.path, 'r')
      dataset = h5fp[class_id]
      dset_length = len(dataset)
      requested = max_buffer_size - current_buffer_length
      end_cursor = cursor + requested
      if end_cursor > dset_length:
        self.cursors[class_id] = requested
        self.buffers[class_id] = list(dataset[:max_buffer_size])
      else:
        self.buffers[class_id].extend(list(dataset[cursor:end_cursor]))
        self.cursors[class_id] = end_cursor
      h5fp.close()

  def process_request(self, request):
    header, data = request
    if header is None:
      return True
    elif isinstance(header, int):
      class_id, amount = data
      indices = np.sort(np.random.permutation(self.buffer_sizes[class_id])[:amount])[::-1]
      self.worker_queues[header].put([self.buffers[class_id].pop(i) for i in indices], block=False)
      self.to_fill.append(class_id)
    return False

  def run(self):
    logging.info("Starting master hdf5 thread")
    self.setup()
    end = False
    while not end:
      for k in self.to_fill:
        self.fill_buffer(k)
      self.to_fill = []
      try:
        request = self.request_queue.get(block=False)
        end = self.process_request(request)
      except queue.Empty:
        pass
    self.h5fp.close()


@gin.configurable(whitelist=["nworkers"])
class SequentialAccessHdf5Backend(BaseBackend):
  """Defines a dataset as a series of h5 files grouped by class in the same
  folder
  """

  def __init__(self, dataset_spec, split, image_size, transforms=None, nworkers=6, fix_missing_images=True):
    """Initializes the hdf5 backend

    Args:
        dataset_spec: an instance from meta_dataset.data.dataset_spec
            describing the input dataset
        image_size: the output image size
        transforms: a function that applies successive transforms to the
            image
        fix_missing_images: the dataset converter sometimes fails to
            read all the images, so the real number of images per class
            is slightly different from the theoretical one.
    """
    self.image_size = image_size
    self.transforms = transforms_lib.Compose([transforms_lib.Lambda(imdecode), transforms])

    self.master_reader = MasterHdf5Reader(dataset_spec,
                                          classes=dataset_spec.get_classes(split),
                                          nworkers=nworkers)
    self.master_reader.start()
    self.worker_queues = self.master_reader.get_worker_queues()
    self.master_queue = self.master_reader.get_request_queue()

    logging.warning(" Ignoring missing images")
    # self.check_missing_images(fix_missing_images)

  def setup(self, worker_id=None):
    """ Thread init function, file pointers are initialized here
        to avoid sharing them between threads

        This should be called manually when not using threading.

    Args:
        worker_id: unique identifier for the thread

    """
    self.id = worker_id
    self.worker_queue = self.worker_queues[worker_id]
    logging.info("Setting up DataLoader worker %d" % self.id)

  def postprocess(self, x):
    """Helper function to ensure that the episode is returned in the correct
    order (samples, channels, h, w)

    Returns: postprocessed episode

    Args:
        x: the episode
    """
    return x

  def read_class(self, class_id, indices):
    """Reads the indexed images from a given class

    Returns: a numpy array with the shape len(indices)xcxhxw

    Args:
        class_id: the class from which to read
        indices: the indices of the images to load
    """
    self.master_queue.put((self.id, (class_id, len(indices))))
    images = self.worker_queue.get(block=True)
    return [self.transforms(im) for im in images]

  def __del__(self):
    if not hasattr(self, "id"):
      self.master_queue.put((None, None))
