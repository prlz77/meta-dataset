import h5py
import numpy as np
import os
import torchvision.transforms as transforms_lib
import logging
import cv2


def imdecode(im):
  return cv2.imdecode(im, cv2.IMREAD_COLOR)



class Backend(object):
  def setup(self):
    raise NotImplementedError

  def read_class(self, class_id, indices):
    raise NotImplementedError


class Hdf5Backend(Backend):
  """Defines a dataset as a series of h5 files grouped by class in the same
  folder
  """

  def __init__(self, dataset_spec, image_size, transforms=None, fix_missing_images=True):
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
    logging.warning(" Ignoring missing images")
    #self.check_missing_images(fix_missing_images)

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
    self.h5fp = h5py.File(self.path, 'r')

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

    dataset = self.h5fp[class_id]

    unique_indices = np.unique(indices)
    sorted_indices = np.sort(unique_indices).tolist()
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
