# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"] 

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0
    
def preprocess_df(df):
    
    df = df.drop('future', 1)
    
    for col in df.columns:
        if col!= "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
            
    df.dropna(inplace=True)
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
            
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys + sells
    random.shuffle(sequential_data)
    
    X = []
    Y = []
    
    for seq, target in sequential_data:
        X.append(seq)
        Y.append(target)
      
    return np.array(X), Y
    
for ratio in ratios:
    dataset = f"/home/kgandhi/Desktop/Deep_Learning/crypto prediction/crypto_data/{ratio}.csv"
    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]    
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]
validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

preprocess_df(main_df)
train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)  

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Train dont buys: {train_y.count(0)} Train buys: {train_y.count(1)}")
print(f"Valid dont buys: {validation_y.count(0)} Valid buys: {validation_y.count(1)}")
