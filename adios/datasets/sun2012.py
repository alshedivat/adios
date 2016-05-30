"""
An interface to the SUN2012 multi-label dataset.
The way the dataset was preprocessed is described in the paper.

Download the original data from:
http://groups.csail.mit.edu/vision/SUN/
"""
import os
import gzip
import logging
import cPickle as pkl

import numpy as np

from core import MLCDataset


class SUN2012(MLCDataset):
    """
    The SUN2012 dataset.

    Parameters
    ----------
    which_set : str
        One of 'train', 'test', or 'full'.
    extended : bool
        whether to use the extended or the original version of the dataset.
    n_labels: int
        number of labels to take when forming the dataset.
    take_labels_from_start: bool
        whether to take the `n_labels` from start or from the end.
    labels_order : str
        One of 'original', 'random', or a name of a *.pkl.gz file.
    min_labels_per_example: int
        minimum number of labels a data point should have; otherwise
        it will be removed from the dataset.
    start : float
        the percentile of the dataset to start from.
    stop : float
        the percentile of the dataset to stop at.
    dataseed : WRITEME
    labelseed : WRITEME
    """
    def __init__(self, which_set, only_labels=False, extended=True,
                 n_labels=None, take_labels_from_start=True,
                 labels_order='original', min_labels_per_example=1,
                 start=0.0, stop=100.0, dataseed=42, labelseed=42):
        # Quick sanity checks
        n_labels = n_labels or 4917
        assert 0 < n_labels <= 4917

        # Dataset parameters
        self.n_features = 1024
        self.n_labels = n_labels
        self.extended = extended

        datadir = os.path.join(os.environ['DATA_PATH'], 'SUN2012')
        super(SUN2012, self).__init__(datadir, which_set,
            n_labels=n_labels,
            only_labels=only_labels,
            labels_order=labels_order,
            take_labels_from_start=take_labels_from_start,
            min_labels_per_example=min_labels_per_example,
            start=start, stop=stop, dataseed=dataseed, labelseed=labelseed)

    def load_dataset(self, which_set):
        if self.verbose:
            sys.stdout.write("Reading data...")

        if which_set not in {'train', 'test', 'full'}:
            raise ValueError(
                'Unrecognized `which_set` value "%s". ' % (which_set,) +
                'Valid values are ["train", "test", "full"].')

        features_filename = which_set + '_features'
        labels_filename = which_set + '_labels'
        if self.extended:
            features_filename += '_ext'
            labels_filename += '_ext'
        features_filename += '.npy'
        labels_filename += '.pkl'

        features_path = os.path.join(self.datadir, features_filename)
        labels_path = os.path.join(self.datadir, labels_filename)

        features = np.load(features_path)
        labels = pkl.load(open(labels_path))
        dataset = {'features': features, 'labels': labels}

        if self.verbose:
            sys.stdout.write("Done.\n")

        return dataset

    def get_design_matrices(self, dataset):
        """
        Take the dataset and extract `X` and `y` design matrices.
        """
        X = np.asarray(dataset['features'], dtype=np.float32)
        y = np.asarray(dataset['labels'].todense(), dtype=np.float32)

        return X, y
