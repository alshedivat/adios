"""
Utility functions for constructing MLC models.
"""
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import ActivityRegularization
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from adios.models import MLC

def assemble(name, params):
    if name == 'MLP':
        return assemble_mlp(params)
    elif name == 'ADIOS':
        return assemble_adios(params)
    else:
        raise ValueError("Unknown name of the model: %s." % name)

def assemble_mlp(params):
    """
    Construct an MLP model of the form:
                                X-H-H1-Y
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """
    model = MLC()

    # X
    model.add_input(name='X', input_shape=(params['X']['dim'],))
    last_layer_name = 'X'

    # H
    if 'H' in params:
        kwargs = params['H']['kwargs'] if 'kwargs' in params['H'] else {}
        model.add_node(Dense(params['H']['dim'], **kwargs),
                       name='H_dense', input=last_layer_name)
        model.add_node(Activation('relu'),
                       name='H_activation', input='H_dense')
        H_output_name = 'H_activation'
        if 'batch_norm' in params['H'] and params['H']['batch_norm'] != None:
            model.add_node(BatchNormalization(**params['H']['batch_norm']),
                           name='H_batch_norm', input=H_output_name)
            H_output_name = 'H_batch_norm'
        if 'dropout' in params['H']:
            model.add_node(Dropout(params['H']['dropout']),
                           name='H_dropout', input=H_output_name)
            H_output_name = 'H_dropout'
        last_layer_name = H_output_name

    # H1
    if 'H1' in params:
        kwargs = params['H1']['kwargs'] if 'kwargs' in params['H1'] else {}
        model.add_node(Dense(params['H1']['dim'], **kwargs),
                       name='H1_dense', input=last_layer_name)
        model.add_node(Activation('relu'),
                       name='H1_activation', input='H1_dense')
        H1_output_name = 'H1_activation'
        if 'batch_norm' in params['H1'] and params['H1']['batch_norm'] != None:
            model.add_node(BatchNormalization(**params['H1']['batch_norm']),
                           name='H1_batch_norm', input=H1_output_name)
            H1_output_name = 'H1_batch_norm'
        if 'dropout' in params['H1']:
            model.add_node(Dropout(params['H1']['dropout']),
                           name='H1_dropout', input=H1_output_name)
            H1_output_name = 'H1_dropout'
        last_layer_name = H1_output_name

    # Y
    kwargs = params['Y']['kwargs'] if 'kwargs' in params['Y'] else {}
    if 'W_regularizer' in kwargs:
      kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    model.add_node(Dense(params['Y']['dim'], **kwargs),
                   name='Y_dense', input=last_layer_name)
    model.add_node(Activation('sigmoid'),
                   name='Y_activation', input='Y_dense')
    Y_output_name = 'Y_activation'
    if 'activity_reg' in params['Y']:
        model.add_node(ActivityRegularization(**params['Y']['activity_reg']),
                       name='Y_activity_reg', input=Y_output_name)
        Y_output_name = 'Y_activity_reg'

    model.add_output(name='Y', input=Y_output_name)

    return model

def assemble_adios(params):
    """
    Construct one of the ADIOS models. The general structure is the following:
                                X-H-(Y0|H0)-H1-Y1,
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """
    model = MLC()

    # X
    model.add_input(name='X', input_shape=(params['X']['dim'],))
    last_layer_name = 'X'

    # H
    if 'H' in params:  # there is a hidden layer between X and Y0
        kwargs = params['H']['kwargs'] if 'kwargs' in params['H'] else {}
        model.add_node(Dense(params['H']['dim'], **kwargs),
                       name='H_dense', input=last_layer_name)
        model.add_node(Activation('relu'),
                       name='H_activation', input='H_dense')
        H_output_name = 'H_activation'
        if 'batch_norm' in params['H'] and params['H']['batch_norm'] != None:
            model.add_node(BatchNormalization(**params['H']['batch_norm']),
                           name='H_batch_norm', input=H_output_name)
            H_output_name = 'H_batch_norm'
        if 'dropout' in params['H']:
            model.add_node(Dropout(params['H']['dropout']),
                           name='H_dropout', input=H_output_name)
            H_output_name = 'H_dropout'
        last_layer_name = H_output_name

    # Y0
    kwargs = params['Y0']['kwargs'] if 'kwargs' in params['Y0'] else {}
    if 'W_regularizer' in kwargs:
      kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    model.add_node(Dense(params['Y0']['dim'], **kwargs),
                   name='Y0_dense', input=last_layer_name)
    model.add_node(Activation('sigmoid'),
                   name='Y0_activation', input='Y0_dense')
    Y0_output_name = 'Y0_activation'
    if 'activity_reg' in params['Y0']:
        model.add_node(ActivityRegularization(**params['Y0']['activity_reg']),
                       name='Y0_activity_reg', input=Y0_output_name)
        Y0_output_name = 'Y0_activity_reg'
    model.add_output(name='Y0', input=Y0_output_name)
    if 'batch_norm' in params['Y0'] and params['Y0']['batch_norm'] != None:
        model.add_node(BatchNormalization(**params['Y0']['batch_norm']),
                       name='Y0_batch_norm', input=Y0_output_name)
        Y0_output_name = 'Y0_batch_norm'

    # H0
    if 'H0' in params:  # we have a composite layer (Y0|H0)
        kwargs = params['H0']['kwargs'] if 'kwargs' in params['H0'] else {}
        model.add_node(Dense(params['H0']['dim'], **kwargs),
                       name='H0_dense', input=last_layer_name)
        model.add_node(Activation('relu'),
                       name='H0_activation', input='H0_dense')
        H0_output_name = 'H0_activation'
        if 'batch_norm' in params['H0'] and params['H0']['batch_norm'] != None:
            model.add_node(BatchNormalization(**params['H0']['batch_norm']),
                           name='H0_batch_norm', input=H0_output_name)
            H0_output_name = 'H0_batch_norm'
        if 'dropout' in params['H0']:
            model.add_node(Dropout(params['H0']['dropout']),
                           name='H0_dropout', input=H0_output_name)
            H0_output_name = 'H0_dropout'
        last_layer_name = [Y0_output_name, H0_output_name]
    else:
        last_layer_name = Y0_output_name

    # H1
    if 'H1' in params:  # there is a hidden layer between Y0 and Y1
        kwargs = params['H1']['kwargs'] if 'kwargs' in params['H1'] else {}
        if isinstance(last_layer_name, list):
            model.add_node(Dense(params['H1']['dim'], **kwargs),
                           name='H1_dense', inputs=last_layer_name)
        else:
            model.add_node(Dense(params['H1']['dim'], **kwargs),
                           name='H1_dense', input=last_layer_name)
        model.add_node(Activation('relu'),
                       name='H1_activation', input='H1_dense')
        H1_output_name = 'H1_activation'
        if 'batch_norm' in params['H1'] and params['H1']['batch_norm'] != None:
            model.add_node(BatchNormalization(**params['H1']['batch_norm']),
                           name='H1_batch_norm', input=H1_output_name)
            H1_output_name = 'H1_batch_norm'
        if 'dropout' in params['H1']:
            model.add_node(Dropout(params['H1']['dropout']),
                           name='H1_dropout', input=H1_output_name)
            H1_output_name = 'H1_dropout'
        last_layer_name = H1_output_name

    # Y1
    kwargs = params['Y1']['kwargs'] if 'kwargs' in params['Y1'] else {}
    if 'W_regularizer' in kwargs:
      kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    if isinstance(last_layer_name, list):
        model.add_node(Dense(params['Y1']['dim'], **kwargs),
                       name='Y1_dense', inputs=last_layer_name)
    else:
        model.add_node(Dense(params['Y1']['dim'], **kwargs),
                       name='Y1_dense', input=last_layer_name)
    model.add_node(Activation('sigmoid'),
                   name='Y1_activation', input='Y1_dense')
    Y1_output_name = 'Y1_activation'
    if 'activity_reg' in params['Y0']:
        model.add_node(ActivityRegularization(**params['Y1']['activity_reg']),
                       name='Y1_activity_reg', input=Y1_output_name)
        Y1_output_name = 'Y1_activity_reg'
    model.add_output(name='Y1', input=Y1_output_name)

    return model
