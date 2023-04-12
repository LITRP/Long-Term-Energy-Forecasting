import sys
import pandas as pd
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

def update_csv():
    df_old = pd.read_csv(f'{data_yml["data"]}')#old data
    df_new = pd.read_csv(f'{data_yml["data_new"]}')#new data
    df_diff = df_new.merge(df_old, how='outer', indicator=True).loc[lambda x : x['_merge']=='left_only']#compare old and new data
    df_diff.to_csv(f'{data_yml["data"]}',index=False)#save new data

def merge_csv():
    df_concat = pd.DataFrame()
    for files in os.listdir(f'to_merge'):#merge all csv files in a folder
        if files.endswith(".csv"):#only csv files
            df = pd.read_csv(f'to_merge/{files}')#read csv
            unwanted_columns  = [x for x in df.columns if x.startswith("Unnamed")]#get unwanted columns
            if len(unwanted_columns) > 0:
                df = df.drop(unwanted_columns,axis=1)#drop unwanted columns
            df_concat = pd.concat([df,df_concat],ignore_index=True)#concatenate
    #df_concat = df_concat.drop_duplicates()#drop duplicates #not needed but can be used if needed
    df_concat.to_csv(f'{data_yml["data"]}',index=False)#save new data


if __name__ == '__main__':
    try:
        if data_yml["update_csv"] == True:#update csv
            update_csv()
    except:
        print("Error in config file update_csv does not exist")
    try:
        if data_yml["merge_csv"] == True:#merge csv
            merge_csv()
    except:

        print("Error in config file merge_csv does not exist")

