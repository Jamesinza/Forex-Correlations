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

    
# ==============================
# 1. Load 1-minute OHLCV data
# ==============================
df = pd.read_csv('datasets/EURUSD_M1_245.csv')
df = df.tail(100_000).copy()
# print(df.head())

# ==============================
# 2. Calculate EMAs and EMA spread
# ==============================
fast_ema_period = 8
slow_ema_period = 21

df['fast_ema'] = df['Close'].ewm(span=fast_ema_period, adjust=False).mean()
df['slow_ema'] = df['Close'].ewm(span=slow_ema_period, adjust=False).mean()
df['ema_spread'] = df['fast_ema'] - df['slow_ema']

# ==============================
# 3. Generate classic crossover signals
# ==============================
df['classic_signal'] = np.where(df['fast_ema'] > df['slow_ema'], 1, -1)
df['actual_signal'] = np.where(df['classic_signal'] != df['classic_signal'].shift(1), df['classic_signal'], 0)

# ==============================
# 4. Feature Engineering
# ==============================
# Returns and volatility
df['returns'] = df['Close'].pct_change()
df['volatility'] = df['returns'].rolling(20).std()

# Volume trend
df['volume_ma'] = df['Volume'].rolling(20).mean()
df['volume_ratio'] = df['Volume'] / df['volume_ma']

# RSI (fixed calculation)
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))

# Drop initial NaNs
df.dropna(inplace=True)

# ==============================
# 5. Labeling logic (target/stop)
# ==============================
lookahead_period = 30  # next 10 bars
target_pct = 0.001     # 0.1% profit
stop_pct = 0.0005      # 0.05% stop-loss

future_max = df['Close'].rolling(lookahead_period).max().shift(-lookahead_period) / df['Close'] - 1
future_min = df['Close'].rolling(lookahead_period).min().shift(-lookahead_period) / df['Close'] - 1

# 1 = target hit first, 0 = stop hit first, ignore otherwise
df['label'] = np.where(future_max > target_pct, 1,
                       np.where(future_min < -stop_pct, 2, 0))

# Keep only actual signal bars for labeling
df['label'] = np.where(df['actual_signal'] == 1, df['label'], 0)
df.dropna(subset=['label'], inplace=True)

# ==============================
# 6. Scaling features
# ==============================
features = ['returns', 'volatility', 'volume_ratio', 'rsi', 'fast_ema', 'slow_ema', 'ema_spread']
scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df[features])
# df[features] = df_scaled
df_features = df[features]

# ==============================
# 7. Sequence creation for LSTM
# ==============================
sequence_length = 30
X_data = df_features.values
y_data = df['label'].values

X = np.empty([len(X_data)-sequence_length, sequence_length, X_data.shape[-1]], dtype=np.float64)
y = np.empty([len(X_data)], dtype=np.int8)
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
print("Class weights:", class_weight_dict)

# ==============================
# 10. LSTM Model
# ==============================
norm = layers.Normalization()
norm.adapt(X_train)

batch_size = compute_batch_size(len(X_train))
dims = 64 #X_train.shape[2]

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    norm,
    layers.LSTM(dims, return_sequences=True),
    layers.Dropout(0.3),
    # layers.LSTM(dims),
    layers.Reshape((X_train.shape[1] * dims,)),
    layers.Dropout(0.3),
    layers.Dense(3, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'] #, keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
)

model.summary()
# ==============================
# 11. Training
# ==============================
callback = [
            callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True, verbose=1),
            # callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, cooldown=20)
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

model.save('test_models/old_school_v2.keras')
# ==============================
# 12. Evaluation
# ==============================
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.7).astype(int)  # threshold = 0.7 for strong signals

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
