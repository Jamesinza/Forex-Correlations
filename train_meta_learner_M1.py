# ==================================================================================================
# ==================================== Training Meta Learner =======================================
# ==================================================================================================

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
from tensorflow.keras import layers, models, callbacks, saving
from sklearn.utils.class_weight import compute_class_weight


@tf.keras.saving.register_keras_serializable()
class ForgetBiasInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        # Bias layout for LSTM: [input_gate, forget_gate, cell_gate, output_gate]
        result = tf.zeros(shape, dtype=dtype)
        n = shape[0] // 4
        result = tf.tensor_scatter_nd_update(result, [[n]], [1.0])
        return result


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


def get_real_data(dataset, num_samples=200_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}_Full.csv')

    col5 = ['Take5','Thunderball','Euro','Mega','Powerball']
    if dataset in col5:
        cols = ['A', 'B', 'C', 'D', 'E'] #['A', 'B', 'C', 'D', 'E']
    else:
        cols = ['A', 'B', 'C', 'D', 'E', 'F']

    if dataset == 'Quick':
        df = df.drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True).astype(np.int8)
    else:
        df = df[cols].dropna().reset_index(drop=True).astype(np.int8)  
        # df = df[df['A'] < 20].dropna().reset_index(drop=True).astype(np.int8)
        # df = df[df['B'] < 30].dropna().reset_index(drop=True).astype(np.int8)
        # df = df[df['D'] > 9].dropna().reset_index(drop=True).astype(np.int8)
        # df = df[df['E'] > 19].dropna().reset_index(drop=True).astype(np.int8)
    
    # Format each element as 2-digit string and then flatten digits into an array.
    df = df.map(lambda x: f'{x:02d}')
    flattened = df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data[-num_samples:]


def compute_batch_size(dataset_length):
    base_unit  = 25000
    base_batch = 32
    batch_size = base_batch * math.ceil(dataset_length / base_unit)
    return batch_size 


def load_base_models(sub_folders, datasets, archs, optimizers, dim, seeds, s, wl):
    for sub_folder in sub_folders:
        for dataset in datasets:
            for arch in archs:
                print(f'\nGetting probabilities from all {arch} models in subfolder {sub_folder} for dataset {dataset} with timestep {wl}...\n')
                for optimizer in optimizers:
                    for seed in seeds:
                        path  = f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}_wl{wl}.keras'
                        model = saving.load_model(path, compile=False)
                        yield model    


# --- Generate 3D data sequences ---
def generate_dataset(X, y, wl=10, features=1):
    X_test = np.empty([len(X)-wl, wl], dtype=np.int8)
    y_test = np.empty([len(X)-wl], dtype=np.int8)
    for i in range(len(X)-wl):
        X_test[i] = X[i:i+wl]
        y_test[i] = y[i+wl]    
    return X_test, y_test


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


def AttBlock(x, num_heads, dim, dropout, seed):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeNormal(seed=seed)
    mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim, seed=seed)(x,x)
    mha = layers.Dropout(dropout, seed=seed)(mha)
    x = layers.LayerNormalization()(x + mha)

    ffn = layers.Dense(dim*4, activation='gelu', kernel_initializer=initializer2)(x)
    ffn = layers.Dropout(dropout, seed=seed)(ffn)
    ffn = layers.Dense(dim, kernel_initializer=initializer1)(ffn)
    ffn = layers.Dropout(dropout, seed=seed)(ffn)
    x = layers.LayerNormalization()(x + ffn)    
    return x


def DenseBlock(x, dim, dropout, seed):
    initializer2 = tf.keras.initializers.HeNormal(seed=seed)
    for mlp in [dim, dim//2, dim//4]:
        x = layers.Dense(mlp, activation='gelu', kernel_initializer=initializer2)(x)
        x = layers.Dropout(dropout, seed=seed)(x)
    return x    

    
def create_meta_learner(meta_arch, n_models, num_classes, dim, optimizer, seed, dropout, num_heads=4, l=1):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeNormal(seed=seed)
    inputs = layers.Input(shape=(n_models, num_classes))
    x = inputs

    if meta_arch=='tcn':
        for i in range(2):
            x = TCNBlock(x,
                         filters=dim,
                         kernel_size=3,
                         dilation_rate=2**i,
                         dropout=dropout,
                         seed=seed,
                        ) 
    elif meta_arch=='dense':
        x = layers.TimeDistributed(layers.Dense(dim, activation="gelu", kernel_initializer=initializer2))(x)
        x = layers.Reshape((x.shape[1]*x.shape[2],))(x)
        for i in range(1):
            x = DenseBlock(x,
                         dim=x.shape[-1],
                         dropout=dropout,
                         seed=seed,
                        )    

    # elif meta_arch=='tsmixer':

    else:
        x = layers.Dense(dim, activation="gelu", kernel_initializer=initializer2)(x)
        for _ in range(2):
            x = AttBlock(x,
                         num_heads=4,
                         dim=dim,
                         dropout=dropout,
                         seed=seed,
                        )

    out = x if meta_arch=='dense' else layers.GlobalAveragePooling1D()(x)  # shape: (batch, 64)

    for units in [dim, dim*2, dim]:
        out = layers.Dense(units, activation='gelu', kernel_initializer=initializer1)(out)
    
    output = layers.Dense(num_classes, kernel_initializer=initializer1)(out)
    model = models.Model(inputs, output, name=f'Meta_L{l}_{meta_arch}_{optimizer}_dim{dim}_seed{seed}')
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["sparse_categorical_accuracy"], jit_compile=True)
    return model


# --- Train Meta Models ---
def train_meta_learner(test_samples, archs, meta_archs, num_classes, batch_size,
                       epochs, datasets, sub_folders, optimizers,
                       dim, seeds, dropout, X_raw, y_data, t_dim, class_weights_dict, targets, wls, l=1):
    idx      = 0
    # p_train  = []
    n_models = len(datasets) * len(optimizers) * len(seeds) * len(archs) * len(sub_folders) * len(wls)
    p_train  = np.empty((test_samples, n_models, num_classes), dtype=np.float32)
    for sub_folder in sub_folders:
        for arch in archs:
            for optimizer in optimizers:
                for wl in wls:
                    X_train, y_train = generate_dataset(X_raw, y_data, wl, 1)
                    target_lst = []
                    for target in targets:
                        print(f'\nLength of training data for wl{wl}: {len(X_train)}\n')
                        path  = f'test_models/{sub_folder}/Take5_{arch}_{optimizer}_dim32_seed42_s{target}_wl{wl}.keras'
                        print(f'\nLoading Model: Take5_{arch}_{optimizer}_dim32_seed42_s{target}_wl{wl}')
                        model = saving.load_model(path, compile=False)
                        res = model.predict(X_train, batch_size=128, verbose=0)
                        res = scaled_logits(res)
                        print(f'\nres shape: {res.shape}')
                        target_lst.append(res[:,1])
                        del model, res, path
                        tf.keras.backend.clear_session()
                    target_lst = np.stack(target_lst, axis=1)
                    
                    # target_lst *= [0.65, 1.0, 1.0, 0.65]
                    
                    print(f'\ntarget_lst shape: {target_lst.shape}\n')
                    p_train[:, idx, :] = target_lst[-test_samples:]
                    idx += 1
                    del target_lst
                    tf.keras.backend.clear_session()

    y_train    = y_train[-test_samples:]
    # p_train    = np.stack(p_train, axis=1)
    # num_models = p_train.shape[1]
    for optimizer in ['adamw']:
        for seed in [42]:
            for meta_arch in meta_archs:
                tf.keras.backend.clear_session()
                callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5)
                ]
                model1 = create_meta_learner(meta_arch, n_models, num_classes, t_dim, optimizer, seed, dropout)
                model1.summary()
                print(f'\nTraining Meta_L{l}_{meta_arch}_{optimizer}_dim{dim}_seed{seed}...')
                model1.fit(
                    p_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1,
                    verbose=1, callbacks=callback, shuffle=False,
                    class_weight=class_weights_dict,
                )
                model1.save(
                    f'test_models/M1_odd/Meta_L{l}_{meta_arch}_{optimizer}_dim{dim}_seed{seed}.keras'
                ) 
                # p_train1 = model1.predict(p_train, verbose=0)
                del model1


def scaled_logits(logits, temperature=2.0):
    scaled_logits = logits / temperature
    scaled_probs  = tf.nn.softmax(scaled_logits)
    return scaled_probs.numpy()             
            

def main():
    seeds        = [42]
    optimizers   = ['adam']  #, 'rmsprop', 'adamw', 'nadam', 'adamax']
    datasets     = ['Take5']
    target_set   = 'Take5'   
    sub_folders  = ['M2_odd']
    archs        = ['tcn','tsmixer','att']
    meta_archs   = ['att','tcn','dense']
    dim          = 32
    dropout      = 0.3
    wls          = [55,89,133,244]  #[8,13,21,34]
    epochs       = 10000
    s            = 0
    targets      = [0,1,2,3]
    test_samples = 58_000
    
    X_raw        = get_real_data(target_set, 120_000)
    X_raw        = X_raw[::2]

    t_dim        = 32
    batch_size   = 32
    
    unique_classes = np.unique(X_raw.flatten())
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
    class_weights_dict = dict(enumerate(class_weights))
    num_classes = len(unique_classes)
    
    meta1, p_train1 = None, None
    tf.keras.backend.clear_session()
    train_meta_learner(test_samples, archs, meta_archs, num_classes, batch_size,
                       epochs, datasets, sub_folders, optimizers,
                       dim, seeds, dropout, X_raw, X_raw, t_dim, class_weights_dict, targets, wls)

if __name__ == '__main__':
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)    

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) 

    main()
