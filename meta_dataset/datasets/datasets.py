import cv2
import h5py
import numpy as np
import torch
import os

from meta_dataset.datasets.episodic_class_dataset import EpisodicClassDataset


class HDF5ClassDataset(EpisodicClassDataset):
    """Defines a dataset as a series of h5 files grouped by class in the same
    folder
    """

    def __init__(self, dataset_spec, split, sampler, image_size, epoch_size, pool=None, reshuffle=True, transforms=None,
                 shuffle_seed=None):
        """Initializes the folder dataset

        Args:
            dataset_spec: an instance from meta_dataset.data.dataset_spec
                describing the input dataset
            split: a meta_dataset.data.learning_spec.Split instance
            sampler: meta_dataset.data.sampling.EpisodeDescriptionSampler
                instance.
            image_size: the output image size
            epoch_size: sets the size of the iterator, precomputes the epoch
                indices
            pool: A string (optional) indicating whether to only read examples
                from a given example-level split.
            reshuffle: whether to reshuffle indices at every epoch
            transforms: a function that applies successive transforms to the
                image
            shuffle_seed:
        """
        super().__init__(dataset_spec, split, sampler, image_size, epoch_size, pool, reshuffle, shuffle_seed)
        self.files_set = os.listdir(dataset_spec.path)
        self.class_names = ["%s.h5" % i for i in self.class_set]
        # Check that the class names correspond to folder names
        assert (set(self.class_names).issubset(self.files_set))
        self.files_set = [os.path.join(dataset_spec.path, path) for path in self.class_names]
        self.transforms = transforms

    def setup(self, worker_id=None):
        """ Thread init function, file pointers are initialized here
            to avoid sharing them between threads

            This should be called manually when not using threading.

        Args:
            worker_id: unique identifier for the thread

        """
        self.h5fp = []
        for path in self.files_set:
            self.h5fp.append(h5py.File(path, 'r'))

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

        Returns: a numpy array with the shape (shot+query)xcxhxw

        Args:
            class_id: the class from which to read
            indices: the indices of the images to load
        """
        if not hasattr(self, "h5fp"):
            raise RuntimeError("self.setup() must be call by each worker, make sure \
                                to check the worker_init_fn in the documentation of the pytorch DataLoader")

        fp_images = self.h5fp[class_id]["images"]
        images = fp_images[sorted(indices)]
        buffer = torch.zeros(len(indices), 3, self.image_size, self.image_size, dtype=torch.get_default_dtype())
        for i, im in enumerate(images):
            im = cv2.imdecode(im, -1)
            buffer[i, ...] = self.transforms(im)

        return buffer

    def __del__(self):
        if hasattr(self, "h5fp"):
            for fp in self.h5fp:
                fp.close()


class FolderClassDataset(EpisodicClassDataset):
    """Defines a dataset as a series of images grouped by class in different
    folders
    """

    def __init__(self, dataset_spec, split, pool, sampler, epoch_size, reshuffle, transforms=None):
        """Initializes the folder dataset

        Args:
            dataset_spec: an instance from meta_dataset.data.dataset_spec
                describing the input dataset
            split: a meta_dataset.data.learning_spec.Split instance
            pool: A string (optional) indicating whether to only read examples
                from a given example-level split.
            sampler: meta_dataset.data.sampling.EpisodeDescriptionSampler
                instance.
            epoch_size: sets the size of the iterator, precomputes the epoch
                indices
            reshuffle: whether to reshuffle indices at every epoch
            transforms: a function that applies successive transforms to the
                image
        """
        super().__init__(dataset_spec, split, pool, sampler, epoch_size, reshuffle)
        self.folders_set = os.listdir(dataset_spec.path)
        self.class_names = [self.dataset_spec.class_names[i] for i in self.class_set]
        # Check that the class names correspond to folder names
        assert (set(self.class_names) in set(self.folders_set))
        self.image_paths = {}
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = lambda x: x
        for class_id in self.class_set:
            path = self.dataset_spec.class_names[class_id]
            root = os.path.join(self.dataset_spec, path)
            ims = os.listdir(root)
            self.image_paths[class_id] = [os.path.join(root, im) for im in ims]

    def postprocess(self, x):
        """Helper function to ensure that the episode is returned in the correct
        order (samples, channels, h, w)

        Returns: postprocessed episode

        Args:
            x: the episode
        """
        return x

    def sample_class(self, class_id, indices):
        """Reads the indexed images from a given class

        Returns: a numpy array with the shape (shot+query)xcxhxw

        Args:
            class_id: the class from which to read
            indices: the indices of the images to load
        """
        images = []
        for i in indices:
            image_path = self.image_paths[class_id][i]
            images.append(self.transform(cv2.imread(image_path)))

        return self.postprocess(images)
