# Silencing annoying warnings
import shutup
shutup.please()

import gc
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, saving, losses
from sklearn.utils.class_weight import compute_class_weight


def get_transformed_data1(dataset, timeframe, session, num_samples=100_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}_{timeframe}_{session}.csv')
    df = df.tail(num_samples).copy()
    col = 'Close' #['A', 'B', 'C', 'D', 'E']
    df = df[[col]]
    df[col] = df[col].pct_change()
    df = df[[col]].dropna().reset_index(drop=True).astype(np.float32)  
    return df.values


def get_transformed_data2(dataset, timeframe, session, num_samples=100_000):
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


# --- TSMixer Model ---
def fnet_tsm_block(x, wl, dim, dropout, features, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)    
    # for _ in range(2):
    #     x = FNetEncoder(dim)(x)
    for _ in range(2):
        x = TSMixerBlock(wl, features, dim, dropout, seed)(x)
    return x


def create_tsmixer_base_model(norm, input_shape, wl, dataset, arch, optimizer,
                              dim, seed, dropout, features):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    inputs = layers.Input(shape=input_shape)
    # x = inputs
    x = inputs  #norm(inputs)
    # x = layers.Dense(dim, activation='gelu', kernel_initializer=initializer1)(x)
    # embed = layers.Embedding(vocab_size, dim)(inputs)
    # pos   = SinePositionEncoding()(embed)
    # x     = embed + pos    

    for _ in range(2):
        x1 = x
        x1 = fnet_tsm_block(x1, wl, dim, dropout, features, seed)
        x = layers.Dropout(dropout)(x)
        x = layers.Add()([x1, x])

    # out =  layers.GlobalAveragePooling1D()(x)
    out =  layers.Reshape((x.shape[1]*x.shape[-1],))(x)
    
    for units in [dim, dim*2, dim]:
        out = layers.Dense(units, activation='gelu', kernel_initializer=initializer1)(out)
        
    output = layers.Dense(1)(out)
    model = models.Model(inputs, output, name=f'{dataset}_{arch}_{optimizer}_dim{dim}')
    model.compile(optimizer=optimizer, loss=losses.Huber(delta=1.0),
                  metrics=['mae','mse'], jit_compile=True)
    return model    


@tf.keras.saving.register_keras_serializable()
class TSMixerBlock(tf.keras.layers.Layer):
    def __init__(self, time_steps=10, num_features=1, hidden_dim=64, dropout_rate=0.1, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.time_steps = time_steps
        self.num_features = num_features

    def build(self, input_shape):
        self.time_dense1 = layers.Dense(
            self.hidden_dim, activation='gelu', kernel_initializer=tf.keras.initializers.HeNormal(seed=self.seed))
        self.time_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        self.time_dense2 = layers.Dense(self.time_steps,
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=self.seed))
        self.time_norm = layers.LayerNormalization()

        self.feature_dense1 = layers.Dense(
            self.hidden_dim, activation='gelu', kernel_initializer=tf.keras.initializers.HeNormal(seed=self.seed))
        self.feature_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        self.feature_dense2 = layers.Dense(self.num_features,
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=self.seed))
        self.feature_norm = layers.LayerNormalization()

        super().build(input_shape)

    def call(self, x):
        # Time-mixing
        residual = x
        x_t = tf.transpose(x, perm=[0, 2, 1])
        x_t = self.time_dense1(x_t)
        x_t = self.time_dropout(x_t)
        x_t = self.time_dense2(x_t)
        x = tf.transpose(x_t, perm=[0, 2, 1])
        x = self.time_norm(x + residual)

        # Feature-mixing
        residual = x
        x_f = self.feature_dense1(x)
        x_f = self.feature_dropout(x_f)
        x_f = self.feature_dense2(x_f)
        x = self.feature_norm(x + residual)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "seed": self.seed,
            "time_steps": self.time_steps,
            "num_features": self.num_features
        })
        return config    


# --- RNN Model ---
@tf.keras.saving.register_keras_serializable()
class ForgetBiasInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        # Bias layout for LSTM: [input_gate, forget_gate, cell_gate, output_gate]
        result = tf.zeros(shape, dtype=dtype)
        n = shape[0] // 4
        result = tf.tensor_scatter_nd_update(result, [[n]], [1.0])
        return result    


def create_rnn_base_model(norm, input_shape, dataset, arch,
                          optimizer, dim, seed, dropout, wl):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeUniform(seed=seed)
    glorot=tf.keras.initializers.GlorotUniform(seed=seed)
    orthog=tf.keras.initializers.Orthogonal(seed=seed)
    bias = ForgetBiasInitializer()
    
    inputs = layers.Input(input_shape)
    x1 = x2 = inputs  #norm(inputs)

    for _ in range(2):
        x1 = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=glorot,
                        recurrent_initializer=orthog)(x1)
        x1 = layers.Dropout(dropout, seed=seed)(x1)
        x1 = layers.LayerNormalization()(x1)
        
    for _ in range(2):
        x2 = layers.LSTM(dim, return_sequences=True, seed=seed, kernel_initializer=glorot,
                         recurrent_initializer=orthog, bias_initializer=bias)(x2)
        x2 = layers.Dropout(dropout, seed=seed)(x2)
        x2 = layers.LayerNormalization()(x2)
    x = layers.Add()([x1, x2])
        
    # out = layers.GlobalAveragePooling1D()(x)
    out =  layers.Reshape((x.shape[1]*x.shape[-1],))(x)
    
    for units in [dim, dim*2, dim]:
        out = layers.Dense(units, activation='gelu', kernel_initializer=initializer1)(out)
        
    output = layers.Dense(1, kernel_initializer=initializer2)(out)
    model = models.Model(inputs, output, name=f'{dataset}_{arch}_{optimizer}_dim{dim}')
    model.compile(optimizer=optimizer, loss=losses.Huber(delta=1.0),
                  metrics=['mae','mse'], jit_compile=False)
    return model     


# --- TCN Model ---
def create_att_tcn_base_model(norm, input_shape, dataset, arch,
                          optimizer, dim, seed, dropout, wl):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    inputs = layers.Input(shape=input_shape)
    x = norm(inputs)

    x = layers.Lambda(lambda z: tf.expand_dims(z, axis=-1), name="expand_feature_dim")(x)  # (B,T,F,1)

    # Shared per-feature embed
    feature_emb = layers.TimeDistributed(
        layers.TimeDistributed(layers.Dense(input_shape[-1], activation="relu")),
        name="shared_feature_mlp"
    )(x)  # (B,T,F,embed_dim)

    # Compute attention logits per feature (shared dense -> scalar per feature)
    attn_logits = layers.TimeDistributed(
        layers.TimeDistributed(layers.Dense(1, activation=None)),
        name="attn_logits"
    )(feature_emb)  # (B,T,F,1)

    # Softmax over features axis (axis=2). We remove the trailing dim for softmax and restore later.
    attn_logits_squeezed = layers.Lambda(lambda z: tf.squeeze(z, axis=-1))(attn_logits)  # (B,T,F)
    attn_weights = layers.Lambda(lambda z: tf.nn.softmax(z, axis=2), name="attn_softmax")(attn_logits_squeezed)  # (B,T,F)
    attn_weights_exp = layers.Lambda(lambda z: tf.expand_dims(z, axis=-1))(attn_weights)  # (B,T,F,1)

    # Weighted sum across features: sum(attn * embedding, axis=2) -> (B,T,embed_dim)
    weighted = layers.Multiply()([feature_emb, attn_weights_exp])
    x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=2), name="attn_pool")(weighted)  # (B,T,embed_dim)

    if input_shape[0] <= 8:
        num_layers = 2
    elif 8 < input_shape[0] <= 34:
        num_layers = 4
    else:
        num_layers = 6
        
    for i in range(num_layers):
        x = TCNBlock(x,
                     filters=dim,
                     kernel_size=3,
                     dilation_rate=2**i,
                     dropout=dropout,
                     seed=seed,
                    )

    out =  layers.Reshape((x.shape[1]*x.shape[-1],))(x)  #GlobalAveragePooling1D()(x)

    # for units in [dim, dim*2, dim]:
    #     out = layers.Dense(units, activation='gelu', kernel_initializer=initializer1)(out)
            
    output = layers.Dense(1)(out)
    model = models.Model(inputs, output, name=f'{dataset}_{arch}_{optimizer}_dim{dim}')
    model.compile(optimizer=optimizer, loss=losses.Huber(delta=0.5),
                  metrics=['mae','mse'], jit_compile=True)
    return model  


def create_tcn_base_model(norm, input_shape, dataset, arch,
                          optimizer, dim, seed, dropout, wl, timeframe):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    inputs = layers.Input(shape=input_shape)
    x = inputs  #norm(inputs)
    # x = layers.Dense(dim, activation='relu')(x)

    # if input_shape[0] <= 8:
    #     num_layers = 2
    # elif 8 < input_shape[0] <= 34:
    #     num_layers = 4
    # else:
    #     num_layers = 6

    kernel_size=3
    rhs = ((input_shape[0] - 1) / (kernel_size - 1)) + 1
    num_layers = math.ceil(math.log2(rhs))    
        
    for i in range(num_layers):
        x = TCNBlock(x,
                     filters=dim,
                     kernel_size=kernel_size,
                     dilation_rate=2**i,
                     dropout=dropout,
                     seed=seed,
                    )

    out =  layers.Reshape((x.shape[1]*x.shape[-1],))(x)
    # out = layers.GlobalAveragePooling1D()(x)

    # for units in [dim]:
    #     out = layers.Dense(units, activation='relu')(out)
            
    output = layers.Dense(1)(out)
    model = models.Model(inputs, output, name=f'{dataset}_{timeframe}_{arch}_{optimizer}_dim{dim}_wl{wl}')
    model.compile(optimizer=optimizer, loss=losses.Huber(delta=0.5),
                  metrics=['mae','mse'], jit_compile=True)
    return model    


def TCNBlock(x, filters, kernel_size, dilation_rate, dropout, seed):
    conv1 = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=dilation_rate)(x)
    conv1 = layers.LayerNormalization()(conv1)
    conv1 = layers.Activation("relu")(conv1)
    conv1 = layers.Dropout(dropout, seed=seed)(conv1)

    conv2 = layers.Conv1D(filters, kernel_size+2, padding="causal",
                          dilation_rate=dilation_rate)(x)
    conv2 = layers.LayerNormalization()(conv2)
    conv2 = layers.Activation("relu")(conv2)
    conv2 = layers.Dropout(dropout, seed=seed)(conv2)

    conv3 = layers.Conv1D(filters, kernel_size+4, padding="causal",
                          dilation_rate=dilation_rate)(x)
    conv3 = layers.LayerNormalization()(conv3)
    conv3 = layers.Activation("relu")(conv3)
    conv3 = layers.Dropout(dropout, seed=seed)(conv3)    

    if x.shape[-1] != filters:
        res = layers.Conv1D(filters, 1, padding="same")(x)
    else:
        res = x

    out = layers.Add()([conv1,conv2,conv3, res])
    return out     


seed=42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

optimizer   = 'adamw'
# corr_data   = 'USDSEK'
session     = 'NoSession'
timeframe   = 'M1'

neg_corr_syms   = ['USDSGD','USDSEK','USDJPY','USDCHF','USDNOK','USDZAR','USDCAD','CADCHF','CADJPY','AUDNZD','AUDJPY','AUDCHF']
# neg_corr_syms.reverse()

pos_corr_syms   = ['EURUSD','GBPUSD','AUDUSD','NZDUSD','EURCAD','NZDCAD','GBPCAD','EURAUD','GBPAUD','AUDCAD','EURCHF','GBPCHF','NZDCHF','GBPJPY']
# pos_corr_syms.reverse()

corr_syms = ['BTCUSD']  #,'EURUSD','AUDUSD','GBPUSD'] #neg_corr_syms + pos_corr_syms
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
    
    plain_data  = None
    # for dataset in datasets:
    get_data   = get_transformed_data1(corr_data, timeframe, session, num_samples=num_samples)
    plain_data = get_data if plain_data is None else np.concatenate([plain_data, get_data], axis=1)
    
    # raw_data     = get_real_data_raw(target_data, timeframe, session, num_samples=100_000)
    # raw_val_data = raw_data[int(len(raw_data)*0.9):]
    
    features    = plain_data.shape[-1]
    input_shape = (wl, features)
    dim         = features
    
    y_data = get_transformed_data1(corr_data, timeframe, session, num_samples=num_samples)
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
    
    norm = tf.keras.layers.Normalization()
    norm.adapt(adapt_ds)
    
    callback = [
                callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
            ]
    
    model = create_tcn_base_model(norm, input_shape, corr_data, arch,
                                      optimizer, dim, seed, dropout, wl, timeframe)
    # model = create_tsmixer_base_model(norm, input_shape, wl, dataset, arch, optimizer,
    #                               dim, seed, dropout, features)
    
    model.summary()
    model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds,
              verbose=1, validation_steps=val_steps, callbacks=callback,
              steps_per_epoch=train_steps) #, class_weight=class_weights_dict)
    model.save(f'test_models/{corr_data}_{timeframe}_{arch}_{optimizer}_dim{dim}_wl{wl}.keras')

    tf.keras.backend.clear_session()