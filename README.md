## Pytorch Version of Meta-Dataset
This repository is a Pytorch port of the original Tensorflow code for reading the Meta-Dataset (https://github.com/google-research/meta-dataset). Arxiv: [https://arxiv.org/abs/1903.03096](https://arxiv.org/abs/1903.03096).

For the sake of reproducibility and development, most of the original code and functionality has been preserved. Only references to tensorflow have been removed or substituted by python built-ins, such as tf.io.gfile -> open.

The current version supports:
- [x] Python3
- [x] Training on ImageNet
- [x] Training on all datasets
- [ ] Fine-grained tests

## Differences with the original code
* TFRecords have been substituted by HDF5 files
* The tensorflow Reader / pipelne has been substituted by a Pytorch [Dataset](https://github.com/prlz77/meta-dataset-pytorch/blob/master/meta_dataset/datasets/class_dataset.py)
* To save time, episode indices are pre-fetch with multiprocessing and cached on disk.

## Basic usage
```python
from meta_dataset.pytorch.meta_dataset import MetaDataset

# Instantiate Meta-Dataset
metatataset = MetaDataset(is_training=True)
# Create training and validation datasets
train_dataset = metatataset.build_dataset("train")
val_dataset = metatataset.build_dataset("valid")

train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=1,
  shuffle=True,
  num_workers=num_workers,
  collate_fn=lambda x: x,
  drop_last=False,
  worker_init_fn=train_dataset.setup, # important, so that threads can initialize correctly
  pin_memory=False)

val_loader = torch.utils.data.DataLoader(
  val_dataset,
  batch_size=1,
  shuffle=False,
  num_workers=num_workers,
  collate_fn=lambda x: x,
  drop_last=False,
  worker_init_fn=val_dataset.setup, # important, so that threads can initialize correctly
  pin_memory=False)
  
for episode in train_loader:
  support_images = episode["support_images"].cuda(async=True)
  query_images = episode["query_images"].cuda(async=True)
```
