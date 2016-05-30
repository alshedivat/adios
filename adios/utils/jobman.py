"""
Utility functions for working with Jobman.
"""
import os
import sys
import yaml

import numpy as np
import itertools as it

from keras.callbacks import EarlyStopping, ModelCheckpoint

from adios.datasets import *
from adios.callbacks import HammingLoss
from adios.metrics import f1_measure
from adios.metrics import hamming_loss
from adios.metrics import precision_at_k
from adios.utils.assemble import assemble_mlp
from adios.utils.assemble import assemble_adios

def expand(d, dict_type=dict):
    """inverse of jobman.tools.flatten()"""
    struct = dict_type()
    for k, v in d.iteritems():
        if k == '':
            raise NotImplementedError()
        else:
            keys = k.split('.')
        current = struct
        for k2 in keys[:-1]:
            current = current.setdefault(k2, dict_type())
        current[keys[-1]] = v  # convert(v)
    return struct

def load_dataset(dataset, labels_order='original'):
    Constructor = getattr(sys.modules[__name__], dataset)
    train = Constructor(which_set='train', stop=80.0,
                        labels_order=labels_order)
    valid = Constructor(which_set='train', start=80.0,
                        labels_order=labels_order)
    test = Constructor(which_set='test', labels_order=labels_order)
    return train, valid, test

def gen_configurations(config_path):
    """
    Read a YAML description of possible configurations search space and
    generate all possible combinations. Yield configuration dictionaries.
    """
    with open(config_path) as fp:
        configs = yaml.load(fp)

    for group in configs:
        # Expand nested dictionaries of lists into lists of dictionaries
        for key, val in group.iteritems():
            if isinstance(val, dict):
                group[key] = [dict(zip(val.keys(), values))
                              for values in it.product(*val.values())]
            elif not isinstance(val, list):
                group[key] = [val]
        # Create the search space (note that it needs to be flattened)
        for values in it.product(*group.itervalues()):
            params = dict(zip(group.iterkeys(), values))
            yield params

def train_mlp(state, channel):
    """
    Train an MLP model specified in the state, and extract required results.
    """
    params = expand(state.parameters)
    os.mkdir('models')

    sys.stdout.write('Loading the data...')
    sys.stdout.flush()

    # Load the data
    train, valid, test = load_dataset(params['dataset'])
    nb_features = train.X.shape[1]
    nb_labels = train.y.shape[1]

    # Convert the datasets into Keras format
    train_data = {'X': train.X, 'Y': train.y}
    valid_data = {'X': valid.X, 'Y': valid.y}
    test_data  = {'X': test.X,  'Y': test.y}

    # Complete the parameters dictionary
    params['X'] = params.get('X', {})
    params['X']['dim'] = nb_features
    params['Y'] = params.get('Y', {})
    params['Y']['dim'] = nb_labels

    sys.stdout.write('Done.\n')
    sys.stdout.write('Assembling & compiling the model...')
    sys.stdout.flush()

    # Assemble the model
    model = assemble_mlp(params)
    model.compile(loss={'Y': 'binary_crossentropy'}, optimizer='adagrad')

    # Save the model configuration
    with open('models/config.yaml', 'w') as fp:
        fp.write(model.to_yaml())

    # Setup callbacks
    callbacks = [
        HammingLoss({'valid': valid_data}),
        ModelCheckpoint('models/best_weights.h5',
                        monitor='val_hl', verbose=0,
                        save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_hl', patience=15, verbose=0, mode='min'),
    ]

    sys.stdout.write('Done.\n')

    # Fit the model
    if train.n_examples < 50000:
        model.fit(train_data, validation_data=valid_data,
                  batch_size=params['batch_size'], nb_epoch=params['nb_epoch'],
                  callbacks=callbacks, verbose=2)
    else:
        batch_gen = train.batch_generator(params['batch_size'],
                                          {'Y': range(nb_labels)})
        model.fit_generator(batch_gen, validation_data=valid_data,
                            samples_per_epoch=train.n_examples,
                            nb_epoch=params['nb_epoch'],
                            callbacks=callbacks, verbose=2)

    # Load the best weights
    model.load_weights('models/best_weights.h5')

    # Fit the thresholds
    model.fit_thresholds(train_data, validation_data=valid_data,
                         alpha=np.logspace(-3, 3, num=10).tolist(),
                         cv=None, verbose=1)

    # Extract results from the trained model
    state.results = extract_results(model, test_data)

    return channel.COMPLETE

def train_adios(state, channel):
    """
    Train an ADIOS model specified in the state, and extract required results.
    """
    params = expand(state.parameters)
    os.mkdir('models')

    sys.stdout.write('Loading the data...')
    sys.stdout.flush()

    # Load the data
    train, valid, test = load_dataset(params['dataset'],
                                      params['labels_order'])
    nb_features = train.X.shape[1]
    nb_labels = train.y.shape[1]

    # Split labels into Y0 and Y1
    nb_labels_per_split = nb_labels // (params['n_label_splits'] + 1)
    nb_labels_Y0 = params['label_split'] * nb_labels_per_split
    nb_labels_Y1 = nb_labels - nb_labels_Y0

    # Convert the datasets into Keras format
    train_data = {'X': train.X,
                  'Y0': train.y[:,:nb_labels_Y0],
                  'Y1': train.y[:,nb_labels_Y0:]}
    valid_data = {'X': valid.X,
                  'Y0': valid.y[:,:nb_labels_Y0],
                  'Y1': valid.y[:,nb_labels_Y0:]}
    test_data = {'X': test.X,
                 'Y0': test.y[:,:nb_labels_Y0],
                 'Y1': test.y[:,nb_labels_Y0:]}

    # Complete the parameters dictionary
    params['X'] = params.get('X', {})
    params['X']['dim'] = nb_features
    params['Y0'] = params.get('Y0', {})
    params['Y0']['dim'] = nb_labels_Y0
    params['Y1'] = params.get('Y1', {})
    params['Y1']['dim'] = nb_labels_Y1

    sys.stdout.write('Done.\n')
    sys.stdout.write('Assembling & compiling the model...')
    sys.stdout.flush()

    # Assemble the model
    model = assemble_adios(params)
    model.compile(loss={'Y0': 'binary_crossentropy',
                        'Y1': 'binary_crossentropy'},
                  optimizer='adagrad')

    # Save the model configuration
    with open('models/config_%d.yaml' % params['label_split'], 'w') as fp:
        fp.write(model.to_yaml())

    # Setup callbacks
    callbacks = [
        HammingLoss({'valid': valid_data}),
        ModelCheckpoint('models/best_weights_%d.h5' % params['label_split'],
                        monitor='val_hl', verbose=0,
                        save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_hl', patience=15, verbose=0, mode='min'),
    ]

    sys.stdout.write('Done.\n')
    sys.stdout.flush()

    # Fit the model
    if train.n_examples < 50000:
        model.fit(train_data, validation_data=valid_data,
                  batch_size=params['batch_size'], nb_epoch=params['nb_epoch'],
                  callbacks=callbacks, verbose=2)
    else:
        batch_gen = train.batch_generator(params['batch_size'],
                                          {'Y0': range(nb_labels_Y0),
                                           'Y1': range(nb_labels_Y0,
                                                       nb_labels)})
        model.fit_generator(batch_gen, validation_data=valid_data,
                            samples_per_epoch=train.n_examples,
                            nb_epoch=params['nb_epoch'],
                            callbacks=callbacks, verbose=2)

    # Load the best weights
    model.load_weights('models/best_weights_%d.h5' % params['label_split'])

    # Fit the thresholds
    model.fit_thresholds(train_data, validation_data=valid_data,
                         alpha=np.logspace(-3, 3, num=10).tolist(),
                         cv=None, verbose=1)

    # Extract results from the trained model
    state.results = extract_results(model, test_data)

    return channel.COMPLETE

def extract_results(model, test_data):
    """
    Return a dictionary of results.
    """
    probs, preds = model.predict_threshold(test_data)

    results = {
        'hl':           hamming_loss(test_data, preds),
        'f1_macro':     f1_measure(test_data, preds, average='macro'),
        'f1_micro':     f1_measure(test_data, preds, average='micro'),
        'f1_samples':   f1_measure(test_data, preds, average='samples'),
        'p@1':          precision_at_k(test_data, probs, K=1),
        'p@3':          precision_at_k(test_data, probs, K=3),
        'p@5':          precision_at_k(test_data, probs, K=5),
        'p@10':         precision_at_k(test_data, probs, K=10),
    }

    return results
