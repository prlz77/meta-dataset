from torch.utils.data import DataLoader
import logging
import time


class EpisodicDataLoader(DataLoader):
    """ Helper wrapping function of the pytorch dataloader.

        Makes sure that Dataset re-randomizes episodes after each "epoch".
    """
    def __iter__(self):
        logging.info("Prefetching episodes")
        t = time.time()
        self.dataset.build_episode_indices()
        logging.info("done in %.01f s" % (time.time() - t))

        return super().__iter__()
