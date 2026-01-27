# Silencing annoying warnings
import shutup
shutup.please()

import gc
import sys
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, saving, optimizers
from keras_hub.layers import FNetEncoder, SinePositionEncoding
from sklearn.utils.class_weight import compute_class_weight


def create_dataset(X, y, W, batch_size, shuffle=False):
    data = np.stack([X, y], axis=1)
    print(f'\n data shape for tf.data: {data.shape}\n')
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(W + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(W + 1))
    dataset = dataset.map(lambda window: (window[:-1, :-1], window[-1, -1]))
    if shuffle==True:
        dataset = dataset.batch(batch_size).shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset    


def prepare_for_adapt(sequence, W):
    dataset = tf.data.Dataset.from_tensor_slices(sequence)
    dataset = dataset.window(W + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(W + 1))
    dataset = dataset.map(lambda window: window[:-1])  # Only inputs
    dataset = dataset.batch(512)  # Batch for efficient adaptation
    return dataset


def process_data(X_train_data, y_train_data, X_val_data, y_val_data,
                 wl, batch_size):
    train_ds = create_dataset(X_train_data, y_train_data, wl, batch_size, shuffle=False)
    val_ds = create_dataset(X_val_data, y_val_data, wl, batch_size, shuffle=False)
    # adapt_ds = prepare_for_adapt(train_data, wl)
    return train_ds, val_ds


def compute_batch_size(dataset_length):
    base_unit  = 25000
    base_batch = 32
    batch_size = base_batch * math.ceil(dataset_length / base_unit)
    return batch_size


@tf.keras.saving.register_keras_serializable()
class ForgetBiasInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        # Bias layout for LSTM: [input_gate, forget_gate, cell_gate, output_gate]
        result = tf.zeros(shape, dtype=dtype)
        n = shape[0] // 4
        result = tf.tensor_scatter_nd_update(result, [[n]], [1.0])
        return result    


def create_rnn_base_model(norm, input_shape, dataset, arch,
                          optimizer, dim, seed, dropout, num_classes, s):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeUniform(seed=seed)
    glorot=tf.keras.initializers.GlorotUniform(seed=seed)
    orthog=tf.keras.initializers.Orthogonal(seed=seed)
    bias = ForgetBiasInitializer()
    
    inputs = layers.Input(input_shape)
    x1 = x2 = norm(inputs)

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
        
    x = layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dense(dim, activation='relu', kernel_initializer=initializer2)(x)
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(x)

    model = models.Model(inputs, out, name=f'{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model    


def cnn_net(inputs, dim, dropout, seed):
    conv1=conv2=conv3=inputs

    conv1 = tf.keras.layers.Conv1D(filters=dim, kernel_size=2, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(conv1)
    conv1 = tf.keras.layers.Dropout(dropout, seed=seed)(conv1)
    conv1 = tf.keras.layers.LayerNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)

    conv2 = tf.keras.layers.Conv1D(filters=dim, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(conv2)
    conv2 = tf.keras.layers.Dropout(dropout, seed=seed)(conv2)
    conv2 = tf.keras.layers.LayerNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)

    conv3 = tf.keras.layers.Conv1D(filters=dim, kernel_size=5, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(conv3)
    conv3 = tf.keras.layers.Dropout(dropout, seed=seed)(conv3)
    conv3 = tf.keras.layers.LayerNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)

    out = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
    return out 


def create_cnn_base_model(norm, input_shape, dataset, arch,
                          optimizer, dim, seed, dropout, num_classes, s):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeUniform(seed=seed)    
    inputs = tf.keras.layers.Input(input_shape)
    
    x = norm(inputs)
    x = cnn_net(x, dim, dropout, seed)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dense(dim, activation='relu', kernel_initializer=initializer2)(x)
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=out, name=f'{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], jit_compile=True)
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

    # def compute_output_shape(self, input_shape):
    #     return input_shape        


def fnet_tsm_block(x, wl, dim, dropout, features, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)    
    # for _ in range(2):
    #     x = FNetEncoder(dim)(x)
    for _ in range(2):
        x = TSMixerBlock(wl, features, dim, dropout, seed)(x)
    return x


def AttBlock(x, num_heads, dim, dropout, seed):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeNormal(seed=seed)
    mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim, seed=seed)(x,x)
    mha = layers.Dropout(dropout, seed=seed)(mha)
    x = layers.LayerNormalization()(x + mha)

    ffn = layers.Dense(dim*4, activation='gelu')(x)
    ffn = layers.Dropout(dropout, seed=seed)(ffn)
    ffn = layers.Dense(dim)(ffn)
    ffn = layers.Dropout(dropout, seed=seed)(ffn)
    x = layers.LayerNormalization()(x + ffn)    
    return x


def create_att_base_model(norm, input_shape, dataset, arch,
                          optimizer, dim, seed, dropout, num_classes, s, wl):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeNormal(seed=seed)
    inputs = layers.Input(shape=input_shape)
    x      = norm(inputs)
    pos    = SinePositionEncoding()(x)
    x      = x + pos
    
    for _ in range(2):
        x = AttBlock(x,
                     num_heads=2,
                     dim=dim,
                     dropout=dropout,
                     seed=seed,
                    )

    out =  layers.GlobalAveragePooling1D()(x)

    for units in [dim, dim*2, dim]:
        out = layers.Dense(units, activation='gelu', kernel_initializer=initializer1)(out)
        
    output = layers.Dense(num_classes)(out)
    model = models.Model(inputs, output, name=f'{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}')
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["sparse_categorical_accuracy"], jit_compile=True)
    return model    

    
# --- TSMixer Model ---
def create_tsmixer_base_model(norm, input_shape, wl, dataset, arch, optimizer,
                              dim, seed, dropout, num_classes, features, s):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    inputs = layers.Input(shape=input_shape)
    # x = inputs
    x = norm(inputs)
    # x = layers.Dense(dim, activation='gelu', kernel_initializer=initializer1)(x)
    # embed = layers.Embedding(vocab_size, dim)(inputs)
    # pos   = SinePositionEncoding()(embed)
    # x     = embed + pos    

    for _ in range(2):
        x1 = x
        x1 = fnet_tsm_block(x1, wl, dim, dropout, features, seed)
        x = layers.Dropout(dropout)(x)
        x = layers.Add()([x1, x])

    out =  layers.GlobalAveragePooling1D()(x)
    
    for units in [dim, dim*2, dim]:
        out = layers.Dense(units, activation='gelu', kernel_initializer=initializer1)(out)
        
    out = layers.Dense(num_classes, kernel_initializer=initializer1)(out)
    model = models.Model(inputs, out, name=f'{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}')
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["sparse_categorical_accuracy"], jit_compile=True)
    return model


# --- TCN Model ---
def create_tcn_base_model(norm, input_shape, dataset, arch,
                          optimizer, dim, seed, dropout, num_classes, s, wl):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    inputs = layers.Input(shape=input_shape)
    x = norm(inputs)

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

    out =  layers.GlobalAveragePooling1D()(x)

    for units in [dim, dim*2, dim]:
        out = layers.Dense(units, activation='gelu', kernel_initializer=initializer1)(out)
            
    out = layers.Dense(num_classes, kernel_initializer=initializer1)(out)
    model = models.Model(inputs, out, name=f'{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}_wl{wl}')
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["sparse_categorical_accuracy"],
                  jit_compile=True,
                 )
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


def trainer(input_shape, norm, train_ds, val_ds, dataset, arch, optimizer, batch_size,
            dim, seed, dropout, epochs, sub_folder, s, num_classes, train_steps, val_steps,
            class_weights_dict, wl, features):
    callback = [
                callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
            ]
    if arch=='tsmixer':
        model = create_tsmixer_base_model(norm, input_shape, wl, dataset, arch,
                                          optimizer, dim, seed, dropout, num_classes, features, s)
    elif arch=='cnn':
        model = create_cnn_base_model(norm, input_shape, dataset, arch,
                                      optimizer, dim, seed, dropout, num_classes, s)
    elif arch=='tcn':
        model = create_tcn_base_model(norm, input_shape, dataset, arch,
                                      optimizer, dim, seed, dropout, num_classes, s, wl)
    else:
        model = create_att_base_model(norm, input_shape, dataset, arch,
                                      optimizer, dim, seed, dropout, num_classes, s, wl)
    
    print(f'\nTraining {dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}_wl{wl}...\n')
    model.summary()
    model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds,
              verbose=1, class_weight=class_weights_dict, validation_steps=val_steps,
              steps_per_epoch=train_steps)
    model.save(f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}_wl{wl}.keras')
    del model, callback
    

def main():
    dataset     = sys.argv[1]
    optimizer   = sys.argv[2]
    sub_folder  = sys.argv[3]
    arch        = sys.argv[4]
    wl          = int(sys.argv[5])
    seed        = int(sys.argv[6])
    s           = int(sys.argv[7])
    features    = 1
    epochs      = 20
    dropout     = 0.3
    dim         = 32
    input_shape = (wl,5)    

    X_train_data = np.load('x_train_data.npy')
    y_train_data = np.load('y_train_data.npy')
    print(f'\nX_train_data: {len(X_train_data)}')
    
    X_val_data = np.load('x_val_data.npy')
    y_val_data = np.load('y_val_data.npy')    
    print(f'\nX_val_data: {len(X_val_data)}\n')

    batch_size = compute_batch_size(len(X_train_data))
    train_steps = math.ceil(len(X_train_data) / batch_size)
    val_steps = math.ceil(len(X_val_data) / batch_size)     

    unique_classes     = np.unique(y_train_data.flatten()) # class counts differ across datasets but max 10
    class_weights      = compute_class_weight('balanced', classes=unique_classes, y=y_train_data.flatten())
    class_weights_dict = dict(enumerate(class_weights))
    num_classes        = len(unique_classes)

    # unique_classes     = np.unique(X_train_data.flatten()) # class counts differ across datasets but max 10
    # vocab_size         = len(unique_classes)    

    train_ds, val_ds = process_data(X_train_data, y_train_data, X_val_data, y_val_data,
                                    wl, batch_size)
    norm = tf.keras.layers.Normalization()
    norm.adapt(X_train_data)
    
    trainer(input_shape, norm, train_ds, val_ds, dataset, arch, optimizer, batch_size,
            dim, seed, dropout, epochs, sub_folder, s, num_classes, train_steps, val_steps,
            class_weights_dict, wl, features
    )

    tf.keras.backend.clear_session()


if __name__ == '__main__':
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed) 

    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    main()
