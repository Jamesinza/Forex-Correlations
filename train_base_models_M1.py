# Silencing annoying warnings
import shutup
shutup.please()

import random
import subprocess
import numpy as np
import pandas as pd


def get_real_data(dataset, num_samples=200_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/{dataset}.csv')
    cols = ['Open', 'High', 'Low', 'Close', 'Volume'] #['A', 'B', 'C', 'D', 'E']
    df = df[cols].dropna().reset_index(drop=True).astype(np.float32)  
    return df.values[-num_samples:]
    

if __name__ == '__main__':
    seed=42
    np.random.seed(seed)
    random.seed(seed)
    
    seeds       = [42]
    optimizers  = ['adam']  #, 'rmsprop', 'adamw', 'nadam', 'adamax']
    datasets    = ['EURUSD_H1'] #,'Thunderball','Euro','Mega','Powerball',
                  # 'C4L','NYLot','HotPicks','Quick']
    time_frame  = 'H1'
    archs       = ['tcn','tsmixer','att'] #,'tsmixer'] #,'cnn','tsmixer','rnn']
    wls         = [8,13,21,34,55] #[8,13,21,34,55]
    sub_folders = ['M1']  #,'M3_odd','M4_odd','M5_odd']
    targets     = [0,1,2]
    # data_seeds  = [6, 28, 42, 276]
    
    for dataset in datasets:
        x_data = get_real_data(dataset)
        
        y_data = np.empty([len(x_data)], dtype=np.int8)
        for i in range(len(x_data)):
            if x_data[i, 0] < x_data[i, 3]:
                y_data[i] = 1  # Bullish candle
                
            elif x_data[i, 0] > x_data[i, 3]:
                y_data[i] = 2  # Bearing candle
                
            else:
                y_data[i] = 0  # Flat candle

        for sub_folder in sub_folders:
            splits = len(x_data)//10
            
            x_train_data = x_data  #[:int(splits*9)]
            y_train_data = y_data  #[:int(splits*9)]
            
            x_val_data = x_data[int(splits*9):]
            y_val_data = y_data[int(splits*9):]
           
            np.save('x_train_data.npy', x_train_data)
            np.save('y_train_data.npy', y_train_data)   
            
            np.save('x_val_data.npy', x_val_data)           
            np.save('y_val_data.npy', y_val_data)     

            print(f'\nx_train_data: {len(x_train_data)}')
            print(f'\nx_val_data: {len(x_val_data)}\n')
            
            for wl in wls:
                for arch in archs:
                    for optimizer in optimizers:
                        for seed in seeds:
                            print(f"\n[LAUNCHER] Starting training in {sub_folder} for {dataset} | {optimizer} | Seed {seed}\n")
                            result = subprocess.run([
                                "python",
                                f"train_worker_{sub_folder}.py",
                                dataset,
                                optimizer,
                                sub_folder,
                                arch,
                                str(wl),
                                str(seed),
                                str(target),
                            ])
                
                            if result.returncode == 0:
                                print(f"[LAUNCHER] Finished training in {sub_folder} for {dataset} | {optimizer} | Seed {seed}\n")
                            else:
                                print(f"[LAUNCHER] Training failed in {sub_folder} for {dataset} | {optimizer} | Seed {seed}\n")
    
    print("\n[LAUNCHER] === ALL TRAINING COMPLETED ===")
