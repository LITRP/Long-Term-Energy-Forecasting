import os
import sys
import warnings

import numpy as np
from prophet import Prophet
from prophet.plot import seasonality_plot_df
from yaml import SafeLoader

from MPTE import file_exists, create_df
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

try: #LOAD CONFIG FILE
    with open(sys.argv[1],encoding="utf-8") as f:
        print(f"Loading config file {sys.argv[1]}")
        data_yml = yaml.load(f, Loader=SafeLoader)
        print("Config file loaded successfully")
        print(f"Archive: {data_yml['data']}")
except: #IF NO FILE IS PASSED AS ARGUMENT
    print("Error loading config file")
    exit()

def Decomposition(m,name,energy): #DECOMPOSITION OF SEASONALITY
    # name = name of the seasonality component you want (yearly,monthly,etc...)
    # m = the model object
    # will return three arrays: days, months, and the seasonality component
    start = pd.to_datetime('2019-01-01 0000')
    period = m.seasonalities[name]['period']
    end = start + pd.Timedelta(days=period)
    plot_points = 366
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    save_dict = {}
    save_dict["per"] = seas[name].values[:-1]
    save_dict['day'] = df_y['ds'].dt.day[:-1]
    save_dict['month'] = df_y['ds'].dt.month[:-1]
    df = pd.DataFrame(save_dict)
    return(save_dict['day'],save_dict['month'],save_dict["per"])

def create_percentage_df(df,energy,df_percentage): #CREATE PERCENTAGE DATAFRAME
    m = Prophet(seasonality_mode='multiplicative')
    df["ds"] = df["ds"].dt.tz_localize(None)#
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    c, a, b = Decomposition(m, "yearly", energy)  #
    df_percentage["month"] = a
    df_percentage["day"] = c
    df_percentage[f"{energy}_per"] = b
    names = pd.DataFrame(df_percentage)
    if data_yml["plot"] == True:
        if not os.path.exists(energy):#CREATE FOLDER
            os.mkdir(energy)#CREATE FOLDER
        m.plot(forecast).savefig(f'{energy}\{energy}_Figure.png')  # plot model
        m.plot_components(forecast).savefig(f"{energy}\{energy}_components.png")  # plot model components
    return names
def fill_df_percentage(df,df_percentage):#FILL PERCENTAGE DATAFRAME
    if not file_exists():#CHECK IF FILE EXISTS
        energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Predict"])))
        energys2 = (list(map(lambda x: x[0], energys)))  # CAN BE BETTER PLEASE CHANGE
        writer = pd.ExcelWriter('Data_Original.xlsx', engine='xlsxwriter')
        for energy in energys2:
            if energy != "Nuclear":
                df2 = df[df[data_yml["tecnology"]] == energy]
                df2 = create_df(df2, energy)
                df2 = df2.rename(columns={data_yml["date"]: "ds", data_yml["generation"]: "y"})
                df2["ds"] = df2["ds"].dt.tz_localize(None)
                df2.to_excel(writer,sheet_name=energy)
                df_percentage = create_percentage_df(df2,energy,df_percentage)
        writer.save()
        df_percentage.to_csv("Percentages.csv", index=False)

if __name__ == '__main__':
    with open(sys.argv[1], "r",encoding="utf-8") as ymlfile:
        data_yml = yaml.safe_load(ymlfile)#LOAD CONFIG FILE
        full_df = pd.read_csv(data_yml["data"])#LOAD DATA
        df_percentage = {}
        fill_df_percentage(full_df, df_percentage)#FILL PERCENTAGE DATAFRAME