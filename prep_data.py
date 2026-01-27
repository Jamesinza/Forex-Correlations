# Silencing annoying warnings
import shutup
shutup.please()

import gc
import math
import random
import subprocess
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, saving
from keras_hub.layers import FNetEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def get_real_data(dataset, num_samples=200_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}.csv')
    df = df.tail(num_samples)

    df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    df = df.set_index("datetime").drop(columns=["Date", "Time"])
    
    cols = ['Open', 'High', 'Low', 'Close', 'Volume'] #['A', 'B', 'C', 'D', 'E']
    # df = df[df['Open']!=df['Close']]
    df = df[cols].dropna().astype(np.float32)
    df["Target"] = np.where(df["Open"] < df["Close"], 1, np.where(df["Open"] > df["Close"], 2, 0))
    df = compute_indicators(df)
    return df.values
    

def compute_indicators(df):
    # # Single series indicators
    df["OC_GAP"] = np.where(df["Open"] < df["Close"].shift(1), 1, np.where(df["Open"] > df["Close"].shift(1), 2, 0))
    # df["EMA_10"] = ta.ema(df["Close"])
    # df["SMA_10"] = ta.sma(df["Close"])
    # df["WMA_10"] = ta.wma(df["Close"])
    # df["TRIMA"]  = ta.trima(df["Close"])
    # df["TEMA"]   = ta.tema(df["Close"])
    # df["SWMA"]   = ta.swma(df["Close"])
    # df["DEMA"]   = ta.dema(df["Close"])
    # df["RSI"]    = ta.rsi(df["Close"])
    # df["ZSCORE"] = ta.zscore(df["Close"])
    # df["BIAS"]   = ta.bias(df["Close"])
    # df["VWMA"]   = ta.vwma(df["Close"],df["Volume"])
    # df["EFI"]    = ta.efi(df["Close"],df["Volume"])
    # df["AO"]     = ta.ao(df["High"],df["Low"])
    # df["CCI"]    = ta.cci(df["High"],df["Low"],df["Close"])
    # df["ATR"]    = ta.atr(df["High"],df["Low"],df["Close"])
    # df["WILLR"]  = ta.willr(df["High"],df["Low"],df["Close"])
    # df["BOP"]    = ta.bop(df["Open"],df["High"],df["Low"],df["Close"])
    # df["VWAP"]   = ta.vwap(df["High"],df["Low"],df["Close"],df["Volume"])
        
    # # Multi series indicators
    # aberrationdf = ta.aberration(df["High"],df["Low"],df["Close"])
    # accbandsdf   = ta.accbands(df["High"],df["Low"],df["Close"])
    # bbandsdf     = ta.bbands(df["Close"])
    # hwcdf        = ta.hwc(df["Close"])
    # donchiandf   = ta.donchian(df["High"],df["Low"])
    # macddf       = ta.macd(df["Close"])
    # adxdf        = ta.adx(df["High"],df["Low"],df["Close"])
    # brardf       = ta.brar(df["Open"],df["High"],df["Low"],df["Close"])
        
    # Candlestick patterns
    cdl_patterndf = ta.cdl_pattern(df["Open"],df["High"],df["Low"],df["Close"])
    cdl_patterndf = cdl_patterndf/100
    df = df[["Target","OC_GAP"]]
    
    df = pd.concat([df, cdl_patterndf], axis=1)  #pd.concat([df,aberrationdf,accbandsdf,bbandsdf,hwcdf,donchiandf,macddf,adxdf,brardf,cdl_patterndf], axis=1)
    df = df.dropna().astype(np.int8)
    return df     
    

if __name__ == '__main__':
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    seeds       = [42]
    optimizers  = ['adam']  #, 'rmsprop', 'adamw', 'nadam', 'adamax']
    datasets    = ['BTCUSD_M15_245'] #,'Thunderball','Euro','Mega','Powerball',
                  # 'C4L','NYLot','HotPicks','Quick']
    time_frame  = 'M1'
    archs       = ['tcn','tsmixer','att'] #,'tsmixer'] #,'cnn','tsmixer','rnn']
    wls         = [8,13,21,34,55] #[8,13,21,34,55]
    sub_folders = ['M1']  #,'M3_odd','M4_odd','M5_odd']
    target      = 0
    # data_seeds  = [6, 28, 42, 276]
    
    for dataset in datasets:
        x_data = get_real_data(dataset, 1_000_000)
        y_data = x_data
        
        # y_data = np.empty([len(x_data), 2], dtype=np.int8)
        # for i in range(len(x_data)):
        #     y_data[i] = x_data[i, [0,3]] #0 if x_data[i, 0] < x_data[i, 3] else 1 # 0=bullish & 1=bearish

        # x_data = y_data  # Ignoring price and only looking at candle type
                
        for sub_folder in sub_folders:
            splits = len(x_data)
            
            x_train_data = x_data[:int(splits-120)]
            y_train_data = y_data[:int(splits-120), target]
            
            x_val_data = x_data[int(splits-120):]
            y_val_data = y_data[int(splits-120):, target]
           
            np.save('x_train_data.npy', x_train_data)
            np.save('y_train_data.npy', y_train_data)   
            
            np.save('x_val_data.npy', x_val_data)           
            np.save('y_val_data.npy', y_val_data)     

            print(f'\nx_train_data shape: {x_train_data.shape}')
            print(f'\nx_val_data shape: {x_val_data.shape}\n')
            
    #         for wl in wls:
    #             for arch in archs:
    #                 for optimizer in optimizers:
    #                     for seed in seeds:
    #                         print(f"\n[LAUNCHER] Starting training in {sub_folder} for {dataset} | {optimizer} | Seed {seed}\n")
    #                         result = subprocess.run([
    #                             "python",
    #                             f"train_worker_{sub_folder}.py",
    #                             dataset,
    #                             optimizer,
    #                             sub_folder,
    #                             arch,
    #                             str(wl),
    #                             str(seed),
    #                             # str(target),
    #                         ])
                
    #                         if result.returncode == 0:
    #                             print(f"[LAUNCHER] Finished training in {sub_folder} for {dataset} | {optimizer} | Seed {seed}\n")
    #                         else:
    #                             print(f"[LAUNCHER] Training failed in {sub_folder} for {dataset} | {optimizer} | Seed {seed}\n")
    
    # print("\n[LAUNCHER] === ALL TRAINING COMPLETED ===")
