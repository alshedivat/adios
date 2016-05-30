"""
An interface to the Delicious multi-label dataset.

Download the dataset from:
http://sourceforge.net/projects/mulan/files/datasets/delicious.rar
"""
import os
import gzip
import logging
import cPickle as pkl

import numpy as np

from core import MLCDataset


class Delicious(MLCDataset):
    """
    The Delicious dataset.

    Parameters
    ----------
    which_set : str
        One of 'train', 'test', or 'full'.
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
    def __init__(self, which_set, only_labels=False,
                 n_labels=None, take_labels_from_start=True,
                 labels_order='original', min_labels_per_example=1,
                 start=0.0, stop=100.0, dataseed=42, labelseed=42):
        # Quick sanity checks
        n_labels = n_labels or 983
        assert 0 < n_labels <= 983

        # Dataset parameters
        self.n_features = 500
        self.n_labels = n_labels

        datadir = os.path.join(os.environ['DATA_PATH'], 'Delicious')
        super(Delicious, self).__init__(datadir, which_set,
            n_labels=n_labels,
            only_labels=only_labels,
            labels_order=labels_order,
            take_labels_from_start=take_labels_from_start,
            min_labels_per_example=min_labels_per_example,
            start=start, stop=stop, dataseed=dataseed, labelseed=labelseed)

    def get_design_matrices(self, dataset):
        """
        Take the dataset and extract `X` and `y` design matrices.
        """
        X = np.asarray(dataset['features'].todense(), dtype=np.float32)
        y = np.asarray(dataset['labels'].todense(), dtype=np.float32)

        return X, y
