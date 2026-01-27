# train_base_models.py

# Silencing annoying warnings
import shutup
shutup.please()

import gc
import math
import random
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, saving
from keras_hub.layers import FNetEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

    col5 = ['Take5','Thunderball','Euro','Mega','Powerball','C4L']
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


def get_real_subdata(dataset, num_samples=200_000, col='A'):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}_Full.csv')

    col5 = ['Take5','Thunderball','Euro','Mega','Powerball']
    if dataset in col5:
        cols = [col] #, 'B', 'C', 'D', 'E']
    else:
        cols = ['A', 'B', 'C', 'D', 'E', 'F']

    if dataset == 'Quick':
        df = df.drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True).astype(np.int8)
    else:
        df = df[cols].dropna().reset_index(drop=True).astype(np.int8)
        # df = df[df[cols] < 30].dropna().reset_index(drop=True).astype(np.int8)
    
    # Format each element as 2-digit string and then flatten digits into an array.
    df = df.map(lambda x: f'{x:02d}')
    flattened = df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data[-num_samples:]


def scaled_logits(logits, temperature=2.0):
    scaled_logits = logits / temperature
    scaled_probs = tf.nn.softmax(scaled_logits)
    return scaled_probs.numpy()      
    

if __name__ == '__main__':
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)     
    
    seeds       = [42]
    optimizers  = ['adam'] #, 'rmsprop', 'adamw', 'nadam', 'adamax']
    dataset     = 'Take5'
    archs       = ['att'] #,'tsmixer','att'] #,'tsmixer','att']
    sub_folder  = 'M1'
    targets     = [0,1,2,3]
    wls         = [8] #,13,21,34,55,89,133,244]  #[89,133,244]
    backtest    = 20 
        
    data    = get_real_data(dataset, 1_000)
    x_data  = data[::2]

    # weights = [0.65, 1.0, 1.0, 0.65]

    for shift in range(backtest*5,4,-5):
        print(f'\n\n\tRunning shift: {shift//5}\n\n')
        for wl in wls:
            b = x_data[:-shift]
            preds_lst = []
            for _ in range(5):
                target_lst = []        
                # for wl in wls:
                input_data = b[-wl:]
                c = x_data if shift==5 else x_data[:-(shift-5)]
                output_data = c[-5:]        
                for arch in archs:
                    for optimizer in optimizers:
                        for target in targets:
                            path  = f'test_models/{sub_folder}/Take5_{arch}_{optimizer}_dim32_seed42_s{target}_wl{wl}.keras'
                            model = saving.load_model(path, compile=False)
                            res1 = model(input_data.reshape(1,wl))
                            res1 = scaled_logits(res1)
                            target_lst.append(res1.flatten()[1])
                            # res2 = np.argmax(res1)
                            # print(f'\nWL: {wl}\tArch: {arch}\tTarget: {target}\tOptimizer: {optimizer}\tRes: {res2.flatten()}\tProb: {res1.flatten()}')
                            tf.keras.backend.clear_session()
    
                target_lst = np.array(target_lst) #* weights # target_lst = np.sum(target_lst, axis=0) #/len(target_lst)
                print(f'\ntarget_lst: {target_lst}')
                res = np.argmax(target_lst)
                preds_lst.append(res)
                b = np.concatenate([input_data, [res]])
                    
            print(f'\nPredictions: {preds_lst}\tTrue Results: {output_data}\n')        
