import sys

import pandas as pd
import numpy as np
import os

import yaml
from yaml import SafeLoader

try: #LOAD CONFIG FILE
    with open(sys.argv[1],encoding="utf-8") as f:
        print(f"Loading config file {sys.argv[1]}")
        data_yml = yaml.load(f, Loader=SafeLoader)
        print("Config file loaded successfully")
except: #IF NO FILE IS PASSED AS ARGUMENT
    print("Error loading config file")
    exit()

if __name__ == '__main__':
    df = pd.read_csv(f'{data_yml["data"]}')
    if data_yml["make_macrozone"] == True:
        for zones in data_yml["macrozones"]:
            df[df["REGIÓN"].isin(data_yml["macrozones"][zones])].to_csv(f'output/{zones}.csv',index=False)
    elif data_yml["make_macrozone"] == False:
        df[data_yml["date"]] = pd.to_datetime(df[data_yml["date"]])
        df[data_yml['generation']] = df[data_yml['generation']].astype('float')
        for x in df["REGIÓN"].value_counts().index:
            df[df["REGIÓN"]==x].to_csv(f'output/{x}.csv',index=False)
    else:
        print("Error in config file")
        exit()

