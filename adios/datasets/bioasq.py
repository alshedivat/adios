"""
An interface to the BioASQ multi-label dataset form Task A, 2015.
The way the dataset was preprocessed is described in the paper.

Download the original data from:
http://participants-area.bioasq.org/

Preprocessed features and labels can be downloaded from the following links:
features (2.3 GB): https://yadi.sk/d/WvwBSg963E8sMq
labels (31 MB): https://yadi.sk/d/zfKrkoDn3E8sDw
"""
import os
import gzip
import logging
import cPickle as pkl

import numpy as np

from core import MLCDataset


class BioASQ(MLCDataset):
    """
    The BioASQ dataset.

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
        n_labels = n_labels or 5000
        assert 0 < n_labels <= 5000

        # Dataset parameters
        self.n_features = 500
        self.n_labels = n_labels

        datadir = os.path.join(os.environ['DATA_PATH'], 'BioASQ')
        super(BioASQ, self).__init__(datadir, which_set,
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
        features_path = os.path.join(self.datadir, which_set + '_features.npy')
        labels_path = os.path.join(self.datadir, which_set + '_labels.pkl')

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
