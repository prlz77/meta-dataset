## Pytorch Version of Meta-Dataset
This repository is a Pytorch port of the original Tensorflow code for reading the Meta-Dataset (https://github.com/google-research/meta-dataset). Arxiv: [https://arxiv.org/abs/1903.03096](https://arxiv.org/abs/1903.03096).

For the sake of reproducibility and development, most of the original code and functionality has been preserved. Only references to tensorflow have been removed or substituted by python built-ins, such as tf.io.gfile -> open.

The current version supports:
- [x] Training on ImageNet
- [x] Training on all datasets
- [ ] Fine-grained tests

## Differences with the original code
* TFRecords have been substituted by HDF5 files
* The tensorflow Reader / pipelne has been substituted by a Pytorch [Dataset](https://github.com/prlz77/meta-dataset-pytorch/blob/master/meta_dataset/datasets/class_dataset.py)
* To save time, episode indices are pre-fetch with multiprocessing and cached on disk.

