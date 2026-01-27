# Silencing annoying warnings
import shutup
shutup.please()

import gc
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, saving, losses, optimizers
from sklearn.utils.class_weight import compute_class_weight


def get_transformed_data_pct(dataset, timeframe, session, num_samples=100_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}_{timeframe}_{session}.csv')
    df = df.tail(num_samples).copy()
    col = 'Close' #['A', 'B', 'C', 'D', 'E']
    df = df[[col]]
    df[col] = df[col].pct_change()
    df = df[[col]].dropna().reset_index(drop=True).astype(np.float64)  
    return df.values


def get_transformed_data_log(dataset, timeframe, session, num_samples=100_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}_{timeframe}_{session}.csv')
    df = df.tail(num_samples).copy()
    col = 'Close' #['A', 'B', 'C', 'D', 'E']
    df = df[[col]]
    # df[col] = df[col].pct_change()
    df[col] = np.log(df[col] / df[col].shift(1))
    df = df[[col]].dropna().reset_index(drop=True).astype(np.float64)  
    return df.values    


def get_transformed_data3(dataset, timeframe, session, num_samples=100_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}_{timeframe}_{session}.csv')
    df = df.tail(num_samples).copy()
    col = 'Close'
    df = df[[col]]
    df[col] = df[col].pct_change()
    df = df[[col]].dropna().reset_index(drop=True).astype(np.float32)
    for lag in [5,8,13,21]:
        df[f'ret_lag_{lag}'] = df[col].shift(lag)
    df = df.dropna().reset_index(drop=True).astype(np.float32)
    return df.values   


def get_real_data_raw(dataset, timeframe, session, num_samples=100_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}_{timeframe}_{session}.csv')
    df = df.tail(num_samples).copy()
    cols = ['Close'] #['A', 'B', 'C', 'D', 'E']
    df = df[cols].dropna().reset_index(drop=True).astype(np.float32)  
    return df.values


def get_real_data_target(dataset, num_samples=200_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}.csv')
    df = df.tail(num_samples).copy()
    cols = ['Open', 'High', 'Low', 'Close'] #['A', 'B', 'C', 'D', 'E']
    df['Target'] = np.where(df['Open'] < df['Close'], 0,
                    np.where(df['Open'] > df['Close'], 1, 2))
    df = df['Target'].dropna().reset_index(drop=True).astype(np.int8)  
    return df.values


def get_real_data_gaps(dataset, num_samples=200_000, target=False, gap=False):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}.csv')
    df = df.tail(num_samples).copy()
    cols = ['Open', 'High', 'Low', 'Close'] #['A', 'B', 'C', 'D', 'E']
    if target:
        df['Target'] = np.where(df['Open'] < df['Close'], 0,
                        np.where(df['Open'] > df['Close'], 1, 2))
        df = df['Target'].dropna().reset_index(drop=True).astype(np.int8)  
    if gap:
        df['Gap'] = np.where(df['Open'] < df['Close'].shift(1), 0,
                        np.where(df['Open'] > df['Close'].shift(1), 1, 2))
        df = df['Gap'].dropna().reset_index(drop=True).astype(np.int8)  
    return df.values    


def create_dataset(data, W, batch_size, shuffle=False):
    # data = np.hstack([X, y])
    print(f'\n data shape for tf.data: {data.shape}\n')
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(W + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(W + 1))
    dataset = dataset.map(lambda window: (window[:-1, :-1], window[-1, -1:]))
    if shuffle==True:
        dataset = dataset.batch(batch_size).shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset     


def prepare_for_adapt(sequence, W):
    dataset = tf.data.Dataset.from_tensor_slices(sequence)
    dataset = dataset.window(W + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(W + 1))
    dataset = dataset.map(lambda window: window[:-1, :-1])  # Only inputs
    dataset = dataset.batch(512)  # Batch for efficient adaptation
    return dataset    


def compute_batch_size(dataset_length):
    base_unit  = 25_000
    base_batch = 32
    batch_size = base_batch * math.ceil(dataset_length / base_unit)
    return batch_size  


# Function to modify the pre-trained model
def finetune_model(input_shape, pretrained_model, dims, dropout, seed):
    inputs = layers.Input(input_shape, name='finetune_inputs')
    pretrained_model(inputs, training=False)
    pre_out = pretrained_model.layers[-3].output  # Get output of layer before final Dense
    
    x = layers.Reshape((pre_out.shape[1]*pre_out.shape[-1],), name='finetune_reshape')(pre_out)
    x = layers.Dropout(dropout, seed=seed, name='finetune_dropout1')(x)
    x = layers.Dense(dims, activation=activation, name='finetune_dense1')(x)    
    outputs = layers.Dense(1, name='finetune_outputs')(x)
    model   = models.Model(pre_out, outputs)
    return model


seed=42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

optimizer      = 'adamw'
# corr_data   = 'USDSEK'
session        = '245'
timeframe      = 'M1'
transformations = ['log', 'pct']
activation     = 'leaky_relu'

neg_corr_syms   = ['USDSGD','USDSEK','USDJPY','USDCHF','USDNOK','USDZAR','USDCAD','CADCHF','CADJPY','AUDNZD','AUDJPY','AUDCHF']
# neg_corr_syms.reverse()

pos_corr_syms   = ['EURUSD','GBPUSD','AUDUSD','NZDUSD','EURCAD','NZDCAD','GBPCAD','EURAUD','GBPAUD','AUDCAD','EURCHF','GBPCHF','NZDCHF','GBPJPY']
# pos_corr_syms.reverse()

corr_syms = ['EURUSD']  #,'EURUSD','AUDUSD','GBPUSD'] #neg_corr_syms + pos_corr_syms
for corr_data in corr_syms:
    # target_data = corr_data
    # datasets    = [corr_data]  #neg_corr_syms + target_data
    arch        = 'tcn'
    wl          = 96
    sub_folder  = 'Correlation_Models'
    target      = 0
    dropout     = 0.5
    # dim         = 32
    epochs      = 1000
    num_samples = 100_000

    for transformation in transformations:
        # plain_data  = None
        # for dataset in datasets:
        get_data   = (get_transformed_data_log(corr_data, timeframe, session, num_samples=num_samples) if transformation=='log'
                      else get_transformed_data_pct(corr_data, timeframe, session, num_samples=num_samples))
        plain_data = get_data.copy() #if plain_data is None else np.concatenate([plain_data, get_data], axis=1)
        
        # raw_data     = get_real_data_raw(target_data, timeframe, session, num_samples=100_000)
        # raw_val_data = raw_data[int(len(raw_data)*0.9):]
        
        features    = plain_data.shape[-1]
        input_shape = (wl, features)
        dim         = features
        
        y_data = get_data.copy()  #get_transformed_data_log(corr_data, timeframe, session, num_samples=num_samples)
        # for i in range(len(plain_data)):
        #     y_data[i] = 1 if plain_data[i, -1] > 0 else 0
        
        # unique_classes     = np.unique(y_data.flatten()) # class counts differ across datasets but max 10
        # class_weights      = compute_class_weight('balanced', classes=unique_classes, y=y_data.flatten())
        # class_weights_dict = dict(enumerate(class_weights))
        # num_classes        = len(unique_classes)    
        
        data = np.concatenate([plain_data, y_data.reshape(-1,1)], axis=1)
        
        train_data = data[:int(len(data)*0.95)]
        val_data   = data[int(len(data)*0.95):]
        
        batch_size  = compute_batch_size(len(train_data))
        
        train_steps = math.ceil(len(train_data) / batch_size)
        val_steps   = math.ceil(len(val_data) / batch_size) 
        
        train_ds    = create_dataset(train_data, wl, batch_size, shuffle=False)
        val_ds      = create_dataset(val_data, wl, batch_size, shuffle=False)
        adapt_ds    = prepare_for_adapt(train_data, wl)
        
        # norm = tf.keras.layers.Normalization()
        # norm.adapt(adapt_ds)
        
        # Stage 1
        callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                ]        
        pretrained_model = saving.load_model((f'test_models/{corr_data}_{timeframe}_{arch}_{optimizer}_dim{dim}_wl{wl}_{transformation}_v2.keras'))
        pretrained_model.trainable=False     # Freeze all layers
        
        model = finetune_model(input_shape, pretrained_model, dim, dropout, seed)
        model.compile(optimizer=optimizers.AdamW(1e-3), loss=losses.Huber(delta=1.0),
                      metrics=['mae','mse'], jit_compile=False)        
        model.summary()
        
        model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds,
                  verbose=1, validation_steps=val_steps, callbacks=callback,
                  steps_per_epoch=train_steps) #, class_weight=class_weights_dict)
        model.save(f'test_models/{corr_data}_{timeframe}_{arch}_{optimizer}_dim{dim}_wl{wl}_{transformation}_v3_stage1.keras')
        tf.keras.backend.clear_session()

        # Stage 2
        callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                ]        
        pretrained_model.trainable=True     # unFreeze all layers
        model.compile(optimizer=optimizers.AdamW(1e-5), loss=losses.Huber(delta=1.0),
                      metrics=['mae','mse'], jit_compile=False)      
        model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds,
                  verbose=1, validation_steps=val_steps, callbacks=callback,
                  steps_per_epoch=train_steps) #, class_weight=class_weights_dict)
        model.save(f'test_models/{corr_data}_{timeframe}_{arch}_{optimizer}_dim{dim}_wl{wl}_{transformation}_v3_stage2.keras')        
        tf.keras.backend.clear_session()