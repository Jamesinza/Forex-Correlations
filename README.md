<h1>Correlation Based Time Series Forecasting</h1>

<p>
This repository provides a modular pipeline for training deep learning models
on short window financial time series. It supports multiple architectures,
including Temporal Convolutional Networks (TCN), TSMixer, and hybrid RNNs,
and is designed for correlated currency pairs, commodities, or crypto data.
</p>

<h2>Features & Highlights</h2>
<ul>
  <li><strong>Flexible Data Loading:</strong> Supports raw prices, percentage changes, lagged returns, and gap/target transformations.</li>
  <li><strong>Time Series Windowing:</strong> Generates sliding windows for input sequences with target alignment.</li>
  <li><strong>Automatic Batch Sizing:</strong> Dynamically computes batch sizes based on dataset length.</li>
  <li><strong>Adaptation Pipeline:</strong> Prepares datasets for normalization and efficient adaptation before model training.</li>
  <li><strong>Multiple Model Architectures:</strong>
    <ul>
      <li>TCN: Multi-scale causal convolutions with dilation, layer normalization, and residual connections.</li>
      <li>TSMixer: Time-mixing + feature mixing blocks inspired by FNet/MLP-Mixer architectures.</li>
      <li>RNN: Stacked GRU + LSTM layers with residual connections, dropout, and normalization.</li>
    </ul>
  </li>
  <li><strong>Training Utilities:</strong> Early stopping, learning rate reduction, mixed precision support, and reproducible seeds.</li>
  <li><strong>Model Saving:</strong> Saves trained models in <code>.keras</code> format for future inference.</li>
</ul>

<h2>Data Pipeline</h2>
<p>
The pipeline handles multiple forms of input:
</p>
<ul>
  <li><strong>Raw data:</strong> Close prices or selected columns, optionally with pct_change transformation.</li>
  <li><strong>Lagged features:</strong> Generates past returns at multiple lags (e.g., 5,8,13,21).</li>
  <li><strong>Targets:</strong> Directional labels (up, down, neutral) or gaps between sessions.</li>
  <li><strong>Windowing:</strong> Sliding windows of size <code>W</code> are extracted for both features and targets.</li>
</ul>

<h2>Model Architectures</h2>
<ul>
  <li><strong>TCNBlock:</strong> Causal convolutions with multiple kernel sizes and dilation rates, residual connections, and dropout.</li>
  <li><strong>TSMixerBlock:</strong> Dense time mixing and feature mixing layers with residual connections and layer normalization.</li>
  <li><strong>RNN Base:</strong> Stacked GRU + LSTM with residual connections, dropout, and layer normalization.</li>
</ul>

<h2>Training Workflow</h2>
<ol>
  <li>Load dataset using one of the <code>get_transformed_data*</code> or <code>get_real_data*</code> functions.</li>
  <li>Concatenate features and targets for training and validation splits.</li>
  <li>Compute batch size automatically based on dataset length.</li>
  <li>Create <code>tf.data.Dataset</code> pipelines with sliding windows and optional shuffling.</li>
  <li>Normalize data using <code>tf.keras.layers.Normalization</code> adapted on training sequences.</li>
  <li>Instantiate model using one of the architecture builders:
    <ul>
      <li><code>create_tcn_base_model</code></li>
      <li><code>create_tsmixer_base_model</code></li>
      <li><code>create_rnn_base_model</code></li>
    </ul>
  </li>
  <li>Train the model with early stopping and learning rate scheduling.</li>
  <li>Save the trained model for inference.</li>
</ol>

<h2>Usage Example</h2>
<pre>
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

dataset = 'BTCUSD'
timeframe = 'M1'
session = 'NoSession'
wl = 96
optimizer = 'adamw'
epochs = 1000
dropout = 0.5

data = get_transformed_data1(dataset, timeframe, session)
y_data = get_transformed_data1(dataset, timeframe, session)
train_data = np.concatenate([data, y_data.reshape(-1,1)], axis=1)
batch_size = compute_batch_size(len(train_data))
train_ds = create_dataset(train_data, wl, batch_size)
</pre>

<h2>Why This Approach?</h2>
<ul>
  <li>TCNs capture long-range temporal dependencies efficiently using causal convolutions and dilation.</li>
  <li>TSMixer leverages time mixing and feature mixing layers to model global interactions in a lightweight way.</li>
  <li>RNN variants provide complementary sequence modeling capabilities.</li>
  <li>Dynamic batch sizing and flexible windowing allow efficient scaling across datasets of different lengths.</li>
  <li>Supports correlation based analysis across multiple assets for multi symbol modeling.</li>
</ul>

<h2>Requirements</h2>
<ul>
  <li>Python 3.9+</li>
  <li>TensorFlow 2.12+</li>
  <li>NumPy, pandas, scikit-learn</li>
  <li><code>shutup</code> (for silencing warnings)</li>
</ul>

<h2>Folder Structure</h2>
<ul>
  <li><code>datasets/</code> : CSV files for each symbol/timeframe/session</li>
  <li><code>test_models/</code> : Trained Keras models will be saved here</li>
  <li><code>scripts/</code> : Optional scripts for training and evaluation</li>
</ul>

<blockquote>
This framework allows experimenting with multiple deep learning architectures for high-frequency time series forecasting, including correlation-aware multi-asset setups. Its modular design makes it easy to extend with new model blocks or input transformations.
</blockquote>
