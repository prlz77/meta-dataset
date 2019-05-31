import torch.multiprocessing as mp
from meta_dataset_pytorch.meta_dataset.datasets.class_dataset import EpisodicClassDataset
import h5py
import os
import logging
import torch
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


def worker_init_fn():
    torch.set_num_threads(1)


def fetch(file, indices, transforms):
    with h5py.File(file, 'r') as h5fp:
        fp_images = h5fp["images"]
        images = fp_images[sorted(indices)]
        buffer = []
        for i, im in enumerate(images):
            buffer.append(transforms(im))
    return torch.stack(buffer, 0)

def _fetch(args):
    return fetch(*args)

class EpisodeProducer(mp.Process):
    def __init__(self, nworkers, episodes, episode_queue, indices, filenames, transforms, return_queue):
        super().__init__(daemon=False)
        self.nworkers = nworkers
        self.transforms = transforms
        self.filenames = filenames
        self.episodes = episodes
        self.episode_queue = episode_queue
        self.return_queue = return_queue
        self.indices = indices

    def get_episode(self, i):
        total_support, total_query, name, episode = self.episodes[i]

        episode["support_images"] = self.buffers[self.current_buffer][0, :total_support]
        episode["query_images"] = self.buffers[self.current_buffer][1, :total_query]
        iterable = zip([self.filenames[int(i)] for i in episode["class_idx"]], episode["indices"], [self.transforms] * len(episode["indices"]))
        images = self.pool.map(_fetch, iterable)

        shot_offset = 0
        query_offset = 0
        for i in range(len(episode["shots"])):
            shot = int(episode["shots"][i])
            query = int(episode["querys"][i])
            im = images[i]
            shot_end = shot_offset + shot
            query_end = query_offset + query
            episode["support_images"][shot_offset:shot_end, ...] = im[:shot]
            episode["query_images"][query_offset:query_end, ...] = im[shot:]
            shot_offset = shot_end
            query_offset = query_end

        return episode

    def run(self):
        self.buffers = self.episode_queue.get(block=True)
        self.current_buffer = 0

        self.pool = mp.Pool(self.nworkers, initializer=worker_init_fn)
        for i in self.indices:
            episode = self.get_episode(i)
            self.return_queue.put(episode, block=True)
            self.current_buffer = (self.current_buffer + 1) % len(self.buffers)


class MultiprocessingEpisodicDataloader(EpisodicClassDataset):
    def __init__(self, dataset_spec, split, sampler, image_size, epoch_size, pool, transforms, nworkers=1, prefetch=2,
                 shuffle=False, reshuffle=True, shuffle_seed=None, fix_missing_images=True):
        super().__init__(dataset_spec, split, sampler, image_size, epoch_size, pool, reshuffle, shuffle_seed)

        self.files_list = os.listdir(dataset_spec.path)
        class_names = [dataset_spec.file_pattern.format(self.class_set[i]) for i in range(len(self.class_set))]
        del self.class_set  # no longer necessary

        # Check that the class names correspond to folder names
        assert (set(class_names).issubset(self.files_list))
        self.files_list = np.array([os.path.join(dataset_spec.path, path) for path in class_names])
        self.transforms = transforms

        logging.warning(" Ignoring missing images")
        self.check_missing_images(fix_missing_images)

        self.prefetch = prefetch
        self.shuffle = shuffle
        self.nworkers = nworkers

    def check_missing_images(self, fix_missing_images):
        """ Checks if the number of examples per class in the dataset spec is the same
            as in the hdf5. If not, it corrects the theoretical number.

        Args:
            fix_missing_images: whether to fail or to fix image counts

        """
        errors = 0
        for i, path in enumerate(self.files_list):
            with h5py.File(path, 'r') as h5fp:
                if h5fp["images"].shape[0] != self.total_images_per_class[i]:
                    self.total_images_per_class[i] = h5fp["images"].shape[0]
                    errors += 1
                if errors > 1 and not fix_missing_images:
                    raise RuntimeError("The number of stored images differs with the count in the DatasetSpec")
        total = int(100 * errors / len(self.files_list))
        if total > 0:
          logging.warning("".join([" {}% of the classes didn't match the theoretical",
                                   " number of examples per class in dataset {}."]).format(total, self.name))

    def fetch_episode(self, item):
        """Reads an episode and returns it

        Returns: a dictionary with the episode

        Args:
            item: episode index in 0..(epoch_size - 1)
        """

        total_support, total_query, episode = self.episodes[item]

        ret = dict(support_images=[],
                   query_images=[],
                   support_class_labels=[],
                   support_episode_labels=[],
                   query_class_labels=[],
                   query_episode_labels=[],
                   shots=[],
                   querys=[], )

        for (class_idx, indices, support_class_labels, query_class_labels,
             support_episode_labels, query_episode_labels, shots, query) in episode:
            ret["support_class_labels"].append(torch.from_numpy(support_class_labels))
            ret["query_class_labels"].append(torch.from_numpy(query_class_labels))
            ret["support_episode_labels"].append(torch.from_numpy(support_episode_labels))
            ret["query_episode_labels"].append(torch.from_numpy(query_episode_labels))
            ret["shots"].append(torch.LongTensor([shots]))
            ret["querys"].append(torch.LongTensor([query]))
            self.input_queues[class_idx].put(indices)

        ret["support_images"] = torch.stack(ret["support_images"], 0)
        ret["query_images"] = torch.stack(ret["query_images"], 0)

        for k in ["support_class_labels", "query_class_labels", "support_episode_labels",
                  "query_episode_labels", "shots", "querys"]:
            ret[k] = torch.cat(ret[k], 0)

        ret["ways"] = len(episode)
        ret["name"] = self.name

        return ret

    def stop_queues(self):
        try:
            self.episode_producer.terminate()
            self.episode_queue.close()
            self.return_queue.close()
            del self.return_queue
            del self.episode_queue
            del self.episode_producer
            del self.buffer
        except:
            pass

    def __iter__(self):
        self.build_episode_indices()

        if hasattr(self, "episode_queue"):
            self.stop_queues()

        self.buffer = [torch.zeros(2, 500, 3, 126, 126) for _ in range(self.prefetch + 2)]
        self.indices = np.random.permutation(self.epoch_size) if self.shuffle else np.arange(self.epoch_size)
        manager = mp.Manager()
        self.episode_queue = manager.Queue(maxsize=self.prefetch)
        self.episode_queue.put(self.buffer)
        self.return_queue = manager.Queue(maxsize=self.prefetch)
        self.episode_producer = EpisodeProducer(self.nworkers, self.episodes, self.episode_queue,
                                                self.indices, self.files_list, self.transforms, self.return_queue)

        self.episode_producer.start()

        self.idx = 0
        return self

    def __next__(self):
        if self.idx > len(self.indices):
            self.stop_queues()
            raise StopIteration
        else:
            ret = self.return_queue.get(block=True)
            self.idx += 1
            return ret

    def __len__(self):
        return self.epoch_size
