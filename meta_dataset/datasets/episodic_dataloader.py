from torch.utils.data import DataLoader


class EpisodicDataLoader(DataLoader):
    """ Helper wrapping function of the pytorch dataloader.

        Makes sure that Dataset re-randomizes episodes after each "epoch".
    """
    def __iter__(self):
        self.dataset.build_episode_indices()
        return super().__iter__()
