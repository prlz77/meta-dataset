import os
import pickle as pkl
import logging
import meta_dataset.data.dataset_spec as dataset_spec_lib

DATASETS_WITH_EXAMPLE_SPLITS = ()

def get_benchmark_specification(dataset_list, records_root_dir, eval_imbalance_dataset, image_shape):
    """Returns a BenchmarkSpecification."""
    valid_benchmark_spec = None  # a benchmark spec for validation only.
    data_spec_list, has_dag_ontology, has_bilevel_ontology = [], [], []
    for dataset_name in dataset_list:
        dataset_records_path = os.path.join(records_root_dir, dataset_name)

        dataset_spec_path = os.path.join(dataset_records_path, 'dataset_spec.pkl')
        if not os.path.exists(dataset_spec_path):
            raise ValueError(
                'Dataset specification for {} is not found in the expected path '
                '({}).'.format(dataset_name, dataset_spec_path))

        with open(dataset_spec_path, 'rb') as f:
            data_spec = pkl.load(f)

        # Replace outdated path of where to find the dataset's records.
        data_spec = data_spec._replace(path=dataset_records_path)

        if dataset_name in DATASETS_WITH_EXAMPLE_SPLITS:
            # Check the file_pattern field is correct now.
            if data_spec.file_pattern != '{}_{}.tfrecords':
                raise RuntimeError(
                    'The DatasetSpecification should be regenerated, as it does not '
                    'have the correct value for "file_pattern". Expected "%s", but '
                    'got "%s".' % ('{}_{}.tfrecords', data_spec.file_pattern))

        logging.info('Adding dataset {}'.format(data_spec.name))
        data_spec_list.append(data_spec)

        # Only ImageNet has a DAG ontology.
        has_dag = False
        if dataset_name == 'ilsvrc_2012':
            has_dag = True
        has_dag_ontology.append(has_dag)

        # Only Omniglot has a bi-level ontology.
        is_bilevel = True if dataset_name == 'omniglot' else False
        has_bilevel_ontology.append(is_bilevel)

        if eval_imbalance_dataset:
            eval_imbalance_dataset_spec = data_spec
            assert len(data_spec_list) == 1, ('Imbalance analysis is only '
                                              'supported on one dataset at a time.')

        # Validation should happen on ImageNet only.
        if dataset_name == 'ilsvrc_2012':
            valid_benchmark_spec = dataset_spec_lib.BenchmarkSpecification(
                'valid_benchmark', image_shape, [data_spec], [has_dag],
                [is_bilevel])

    benchmark_spec = dataset_spec_lib.BenchmarkSpecification(
        'benchmark', image_shape, data_spec_list, has_dag_ontology,
        has_bilevel_ontology)

    return benchmark_spec, valid_benchmark_spec

