import math
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt

seed=42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


def compute_batch_size(dataset_length):
    base_unit  = 25_000
    base_batch = 32
    batch_size = base_batch * math.ceil(dataset_length / base_unit)
    return batch_size


def create_labeled_feature_df(dataset: str='datasets/EURUSD_M1_245.csv', num_samples: int=10_000, lookback: int=30,
                              lookahead: int=15, min_points: int=50) -> pd.DataFrame:
    """
    Create a dataset with OHLC + engineered features + labels for EURUSD strategy.
    
    Labels:
        0 = no trade
        1 = buy
        2 = sell
    
    Features: strictly causal (no future leakage).
    """

    raw_df = pd.read_csv(dataset)
    df = raw_df.tail(num_samples).copy().reset_index(drop=True)

    # # Getting rid of bad trading periods
    # df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    # df["hour"] = df["datetime"].dt.hour
    # df = df[~df["hour"].isin([0,1,3,4,5,6,7,8,9,10,11,12, 18,19,20,21,22,23])]
    # df = df.drop(columns=["hour"]).reset_index(drop=True)
    
    df = df[['Open','High','Low','Close','Volume']]
    point = 0.00001
    threshold = min_points * point
    
    # --- LABEL CREATION ---
    labels = []
    for i in range(len(df) - lookahead):  # stop early to avoid future leakage
        current_close = df.loc[i, "Close"]
        future_high = df.loc[i+1:i+lookahead, "High"].max()
        future_low = df.loc[i+1:i+lookahead, "Low"].min()
        
        diff_high = abs(future_high - current_close)
        diff_low = abs(future_low - current_close)
        
        if diff_high > diff_low and diff_high >= threshold:
            labels.append(1)  # buy
        elif diff_high < diff_low and diff_low >= threshold:
            labels.append(2)  # sell
        else:
            labels.append(0)  # no trade
    
    # truncate df to valid rows
    df = df.iloc[:len(df) - lookahead].copy()
    df["Label"] = labels
    
    # --- FEATURE ENGINEERING ---
    
    # 1. Relative Close vs Range
    df["rel_close"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

    # 2. Exponential Moving Averages and Momentum
    df["ma_fast"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["ma_slower"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["ma_slow"] = df["Close"].ewm(span=13, adjust=False).mean()
    df["ma_diff1"] = df["ma_fast"] - df["ma_slower"]
    df["ma_diff2"] = df["ma_slower"] - df["ma_slow"]

    # 3. RSI
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(lookback).mean()
    roll_down = pd.Series(loss).rolling(lookback).mean()
    RS = roll_up / (roll_down + 1e-9)
    df["RSI"] = 100 - (100 / (1 + RS))

    # 4. ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(lookback).mean() / point

    # 5. Candle Body Size
    df["body_size"] = (df["Close"] - df["Open"]).abs() / point

    # 6. Upper/Lower Wick Ratios
    df["upper_wick"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / (df["High"] - df["Low"] + 1e-9)
    df["lower_wick"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

    # 7. Volatility Compression
    df["vol_comp"] = (df["High"] - df["Low"]).rolling(lookback).std() / point

    # 8. Lookback Breakout Features
    df["lookback_high"] = df["High"].rolling(lookback).max()
    df["lookback_low"] = df["Low"].rolling(lookback).min()
    df["dist_to_high"] = (df["lookback_high"] - df["Close"]) / point
    df["dist_to_low"] = (df["Close"] - df["lookback_low"]) / point

    # 9. Returns and volatility
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # 10. Volume trend
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']    
    
    # final cleanup: drop NaNs from rolling windows
    df = df.dropna().reset_index(drop=True)
    
    return df    

    
# ==============================
# 1. Load 1-minute OHLCV data
# ==============================
# df = pd.read_csv('datasets/EURUSD_M1_245.csv')
# df = df.tail(100_000).copy()
# print(df.head())
sequence_length = 15
dataset = 'datasets/EURUSD_M1_245.csv'
data = create_labeled_feature_df(dataset, 100_000, sequence_length, lookahead=15)
print(f'\nData shape: {data.shape}\n')

X_data = data.drop(columns=['Label']).values
y_data = data['Label'].values

X = np.empty([len(X_data)-sequence_length, sequence_length, X_data.shape[-1]], dtype=np.float64)
y = np.empty([len(X_data)-sequence_length], dtype=np.int8)
for i in range(len(X_data)-sequence_length):
    X[i] = X_data[i:i+sequence_length]
    y[i] = y_data[i+sequence_length]

# X = np.array(X)
# y = np.array(y)

# ==============================
# 8. Train/Test split (chronological)
# ==============================
split_idx = int(0.9 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ==============================
# 9. Compute class weights to handle imbalance
# ==============================
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict =  dict(zip(classes.astype(int), class_weights))
num_classes = len(classes)
print("Class weights:", class_weight_dict)

# ==============================
# 10. LSTM Model
# ==============================
norm = layers.Normalization()
norm.adapt(X_train)

batch_size = compute_batch_size(len(X_train))
dims = X_train.shape[2]

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    norm,
    layers.GRU(dims, return_sequences=True),
    layers.GRU(dims, return_sequences=True),
    layers.LSTM(dims, return_sequences=True),
    layers.Dropout(0.1),
    # layers.LSTM(dims),
    layers.Reshape((X_train.shape[1] * dims,)),
    # layers.Dropout(0.1),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adamw',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'] #, keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
)

model.summary()
# ==============================
# 11. Training
# ==============================
callback = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, cooldown=0)
        ] 

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    callbacks=callback,
    shuffle=False,
    epochs=10_000,
    batch_size=batch_size,
    class_weight=class_weight_dict,
    verbose=1
)

model.save('test_models/old_school_v3.keras')
# # ==============================
# # 12. Evaluation
# # ==============================
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.show()

y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1) #(y_pred_proba > 0.7).astype(int)  # threshold = 0.7 for strong signals

print(classification_report(y_test, y_pred))

# ==============================
# 13. Real-time use notes
# ==============================
# For each new bar:
# 1. Update features (returns, EMA, RSI, volume_ratio, etc.)
# 2. Check for crossover signal
# 3. Form last `sequence_length` bars, scale using the same `scaler`
# 4. Predict probability
# 5. Trade only if prob > threshold (0.7)
# 6. Use tight stop-loss / take-profit as defined
