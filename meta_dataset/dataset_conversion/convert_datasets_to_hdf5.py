# coding=utf-8
# Copyright 2019 The Meta-Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyformat: disable
r"""Main file for converting the datasets used in the benchmark into records.

Example command to convert dataset omniglot:
# pylint: disable=line-too-long
python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=omniglot \
  --omniglot_data_root=<path/to/omniglot> \
  --records_root=<path/to/records> \
  --splits_root=<path/to/splits>
# pylint: enable=line-too-long
"""
# pyformat: enable
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
logging.getLogger().setLevel(logging.INFO)

from meta_dataset.dataset_conversion import dataset_to_hdf5
from meta_dataset.utils.argparse import argparse

parser = argparse.parser
parser.add_argument('--mini_imagenet_records_dir', type=str, default='',
                    help='The path to store the tf.Records of MiniImageNet.')
parser.add_argument('--dataset', type=str, default='omniglot',
                    help='The name of the dataset to convert to records.')
FLAGS = argparse.FLAGS


def main():
    class ConverterArgs(
        collections.namedtuple('ConverterArgs', 'data_root, long_name')):
        """Arguments to be passed to a DatasetConverter's constructor.

        Args:
          data_root: string, path to the root of the dataset.
          long_name: string, dataset name in longer or capitalized form.
        """

    # Maps dataset name to (converter class, arguments) pairs.
    # This will be used to build the corresponding DatasetConverter object.
    dataset_name_to_converter_and_args = {
        # Datasets in the same order as reported in the article.
        'ilsvrc_2012': (dataset_to_hdf5.ImageNetConverter,
                        ConverterArgs(
                            data_root=FLAGS.ilsvrc_2012_data_root,
                            long_name='ImageNet ILSVRC-2012')),
        'omniglot': (dataset_to_hdf5.OmniglotConverter,
                     ConverterArgs(
                         data_root=FLAGS.omniglot_data_root,
                         long_name='Omniglot')),
        'aircraft': (dataset_to_hdf5.AircraftConverter,
                     ConverterArgs(
                         data_root=FLAGS.aircraft_data_root,
                         long_name='FGVC-Aircraft Benchmark')),
        'cu_birds': (dataset_to_hdf5.CUBirdsConverter,
                     ConverterArgs(
                         data_root=FLAGS.cu_birds_data_root,
                         long_name='CU Birds')),
        'dtd': (dataset_to_hdf5.DTDConverter,
                ConverterArgs(
                    data_root=FLAGS.dtd_data_root,
                    long_name='Describable Textures Dataset')),
        'quickdraw': (dataset_to_hdf5.QuickdrawConverter,
                      ConverterArgs(
                          data_root=FLAGS.quickdraw_data_root,
                          long_name='Quick, Draw!')),
        'fungi': (dataset_to_hdf5.FungiConverter,
                  ConverterArgs(
                      data_root=FLAGS.fungi_data_root,
                      long_name='fungi 2018 FGVCx')),
        'vgg_flower': (dataset_to_hdf5.VGGFlowerConverter,
                       ConverterArgs(
                           data_root=FLAGS.vgg_flower_data_root,
                           long_name='VGG Flower')),
        'traffic_sign': (dataset_to_hdf5.TrafficSignConverter,
                         ConverterArgs(
                             data_root=FLAGS.traffic_sign_data_root,
                             long_name='Traffic Sign')),
        'mscoco': (dataset_to_hdf5.MSCOCOConverter,
                   ConverterArgs(
                       data_root=FLAGS.mscoco_data_root, long_name='MSCOCO')),
        # Diagnostics-only dataset
        'mini_imagenet': (dataset_to_hdf5.MiniImageNetConverter,
                          ConverterArgs(
                              data_root=FLAGS.mini_imagenet_data_root,
                              long_name='MiniImageNet')),
    }

    if FLAGS.dataset not in dataset_name_to_converter_and_args:
        raise NotImplementedError(
            'Dataset {} not supported. Supported datasets are {}'.format(
                FLAGS.dataset, sorted(dataset_name_to_converter_and_args.keys())))

    converter_class, converter_args = dataset_name_to_converter_and_args[
        FLAGS.dataset]
    if FLAGS.dataset == 'mini_imagenet':
        # MiniImagenat is for diagnostics purposes only, do not use the default
        # records_path to avoid confusion.
        records_path = FLAGS.mini_imagenet_records_dir
    else:
        records_path = None
    converter = converter_class(
        name=FLAGS.dataset,
        data_root=converter_args.data_root,
        records_path=records_path)
    logging.info(
        'Creating {} specification and records in directory {}...'.format(
            converter_args.long_name, converter.records_path))
    converter.convert_dataset()


if __name__ == '__main__':
    argparse.parser.parse_args()
    main()
