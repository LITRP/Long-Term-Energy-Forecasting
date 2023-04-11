# -*- coding: utf-8 -*-
'''
author: @jsotoL
created: 01-03-2023
version: 1.5
'''

import warnings

import numpy as np
import pandas as pd
import sys


import math_
import matplotlib.pyplot as plt
import logging
import yaml
import prophet
from yaml.loader import SafeLoader

import os

from tool import combine_dic

os.environ['KMP_DUPLICATE_LIB_OK']='True' #HACK TO MAKE IT WORK


try: #LOAD CONFIG FILE
    with open(sys.argv[1],encoding="utf-8") as f:
        print(f"Loading config file {sys.argv[1]}")
        data_yml = yaml.load(f, Loader=SafeLoader)
        print("Config file loaded successfully")
        print(f"Archive: {data_yml['data']}")
except: #IF NO FILE IS PASSED AS ARGUMENT
    print("Error loading config file")
    exit()




logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)

import sys,time,random


class timeseries(): #CLASS TIMESERIES
    def __init__(self): # INIT NEW TIMESERIES
        self.data = pd.DataFrame()
        self.name = "timeseries"
        self.units = "units"
        self.time_units = "time units"
        logging.info(f"{time.ctime(int(time.time()))} New timeseries created")

    def add(self, time, data): #ADD DATA MANUALLY TO TIMESERIES
        self.data["y"].append(data)
        self.data["ds"].append(time)

    def print(self): #PRINT TIMESERIES
        for i in range(len(self.data)):
            print(self.time[i], self.data[i])

    def plot(self): #PLOT TIMESERIES
        plt.plot(self.data["ds"], self.data["y"])
        plt.xlabel(self.time_units)
        plt.ylabel(self.units)
        plt.title(self.name)
        plt.show()
    def save(self): #SAVE TIMESERIES TO CSV
        try:
            df = pd.DataFrame(self.data, index=self.time, columns=[self.name])
            df.to_csv(self.name.replace(" ","") + ".csv") #SAVE TO CSV FILE using name of timeseries
            logging.info(f"{time.ctime(int(time.time()))} File saved successfully")
        except:
            print("Error saving file")
    def read_csv(self,filename): #READ CSV FILE
        try:
            df = pd.read_csv(filename)
            self.data = df[data_yml["generation"]]
            self.time = df.index
            logging.info(f"{time.ctime(int(time.time()))} File read successfully")
        except:
            print("Error reading file")
            logging.error(f"{time.ctime(int(time.time()))} Error reading file in {self.name}")
    def convert_to_dataframe(self): #CONVERT TIMESERIES TO DATAFRAME
        data = self.data
        df = pd.DataFrame(data)
        df["ds"] = pd.to_datetime(self.data["ds"])
        return df


def save_excel(df,name,writer): #SAVE TIMESERIES TO EXCEL
    try:
        df.to_excel(writer, sheet_name=name)
        logging.info(f"{time.ctime(int(time.time()))} File saved successfully")
    except Exception as e:
        print("Error saving file")
        #print(e)
        logging.error(f"{time.ctime(int(time.time()))} Error saving file in {name}")


def prophet_model_predict(df,fechas): #PROPHET MODEL
    model = prophet.Prophet()
    model.fit(df)
    period = len(fechas) - len(df)
    future = model.make_future_dataframe(periods=period, freq="D")
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat"]]
    forecast = forecast.rename(columns={"ds": "ds", "yhat": "y"})
    return forecast
def Total_timeseries(df3): #MAKE TOTAL TIMESERIES:
    df3 = df3.groupby(by=["Date"])["Generacion_MWh"].sum().reset_index()
    df3 = df3[["Date", "Generacion_MWh"]]
    Serie = timeseries()
    Serie.name = "Total"
    Serie.units = "MWh"
    Serie.time_units = "Date"
    Serie.data["y"] = df3["Generacion_MWh"]
    Serie.data["ds"] = df3["Date"]
    return Serie


def make_renowable_timeseries(df): #MAKE RENEWABLE TIMESERIES:
    energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Renewable"]))) #GET RENEWABLE ENERGYS
    energys2 = (list(map(lambda x: x[0], energys))) #GET RENEWABLE ENERGYS 2
   # print(energys2)
    df = df[df[data_yml["tecnology"]].isin((energys2))]
    df = df.groupby(by=["Date"])["Generacion_MWh"].sum().reset_index()
    df = df[["Date", "Generacion_MWh"]]
    Serie = timeseries()
    Serie.name = "Renowable"
    Serie.units = "MWh"
    Serie.time_units = "Date"
    Serie.data["y"] = df["Generacion_MWh"]
    Serie.data["ds"] = df["Date"]
    return Serie

def make_fossil_timeseries(df): #MAKE FOSSIL TIMESERIES:
    energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Fossil"]))) #GET FOSSIL ENERGYS
    energys2 = (list(map(lambda x: x[0], energys))) #GET FOSSIL ENERGYS 2
    #print(energys2)
    df = df[df[data_yml["tecnology"]].isin((energys2))]
    df = df.groupby(by=["Date"])["Generacion_MWh"].sum().reset_index()
    df = df[["Date", "Generacion_MWh"]]
    Serie = timeseries()
    Serie.name = "Fossil"
    Serie.units = "MWh"
    Serie.time_units = "Date"
    Serie.data["y"] = df["Generacion_MWh"]
    Serie.data["ds"] = df["Date"]
    return Serie





def get_percentage_renew():
    energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Predict"]))) # CAN BE BETTER PLEASE CHANGE
    energys2 = (list(map(lambda x: x[0], energys))) # CAN BE BETTER PLEASE CHANGE
    perce = (list(map(lambda x: list(x.values()), data_yml["energy"]["Predict"]))) # CAN BE BETTER PLEASE CHANGE
    perce2 = (list(map(lambda x: x[0], perce))) #PERCENTAGE # CAN BE BETTER PLEASE CHANGE
    diccionario = dict(zip(energys2, perce2))
    return diccionario


def get_centrals(energy):
    energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Predict"])))  # CAN BE BETTER PLEASE CHANGE
    energys2 = (list(map(lambda x: x[0], energys)))  # CAN BE BETTER PLEASE CHANGE
    perce = (list(map(lambda x: list(x.values()), data_yml["energy"]["Predict"])))  # CAN BE BETTER PLEASE CHANGE
    perce2 = (list(map(lambda x: x[0], perce)))  # PERCENTAGE # CAN BE BETTER PLEASE CHANGE
    diccionario = dict(zip(energys2, perce2))
    return (diccionario[energy][2]["data"]),(diccionario[energy][3]["values"])

def get_type(energy):
    energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Predict"])))  # CAN BE BETTER PLEASE CHANGE
    energys2 = (list(map(lambda x: x[0], energys)))  # CAN BE BETTER PLEASE CHANGE
    perce = (list(map(lambda x: list(x.values()), data_yml["energy"]["Predict"])))  # CAN BE BETTER PLEASE CHANGE
    perce2 = (list(map(lambda x: x[0], perce)))  # PERCENTAGE # CAN BE BETTER PLEASE CHANGE
    diccionario = dict(zip(energys2, perce2))
    return str(diccionario[energy][1])
def create_df(df,energy): #CREATE DF
    #print(df)
    df[data_yml["date"]] = pd.to_datetime(df[data_yml["date"]],utc=True)
    df = df.groupby(by=[data_yml["date"]])[data_yml["generation"]].sum().reset_index()
    return df


def create_total(df,name): #SUM ALL COLUMNS
    pivot = df.sum(axis=1)
    pivot_DF = pd.DataFrame(pivot)
    df[f"Total_{name}"] = pivot_DF.sum(axis=1)
    return df

from prophet.plot import seasonality_plot_df

def get_index(date_list, date): #GET INDEX FROM DATE

    return list(date_list).index(date)

def Javier_Regression(full_df,df,percentage,energy,df_percentage,type): #JAVIER REGRESSION FUNCTION
    if not file_exists():
        print("No existe el archivo de porcentajes nesesario para la predicción")
        print("Cerrando programa")
        exit()
    else:
        names = pd.read_csv(f"output/Percentages.csv")
    if type =="Demand":
        fechas = pd.date_range(start=full_df["ds"].max().replace(tzinfo=None), end=data_yml["objetive_date"], freq=data_yml["freq"])
        pen, b = math_.ecuación_de_la_recta(len(full_df), df["y"][df["ds"] == full_df["ds"].max().replace(tzinfo=None)].iloc[0], len(df) + len(fechas),
                                        data_yml["target_production"] * percentage)
    else:
        fechas = pd.date_range(start=df["ds"].max().replace(tzinfo=None), end=data_yml["objetive_date"], freq=data_yml["freq"])
        pen, b = math_.ecuación_de_la_recta(len(df), df["y"].iloc[-1], len(df)+len(fechas), data_yml["target_production"]*percentage)
    lista = []
    offset = len(df)
    x = np.arange(offset, offset+len(fechas), 1)
    if type == "linear" or type == "Demand":
        result = math_.calcular_puntos_de_la_recta(pen, b, len(df), x)
        lista.extend(result)
    else:#x, a, b, c
        def pendiente(x1, x2, y1, y2):
            return ((y2 - y1) / (x2 - x1))

        def indep(x1, y1, m):
            return (y1 - (m * x1))

        def puntos(m, x, b):
            return ((m * x) + b)

        date_list, values_list = get_centrals(energy)
        index = []
        pivot = fechas.tolist()
        for d in date_list:
            if d in pivot:
                index.append(get_index(pivot, d))
        #print(index)
        pen = pendiente(0, len(fechas), df["y"].iloc[-1], data_yml["target_production"]*percentage)
        b = indep(0, df["y"].iloc[-1], pen)
        listo_y = list(map(lambda x: puntos(pen, x, b), range(0, len(fechas))))
        filtro = list(filter(lambda x: x in index, range(0, len(fechas))))
        for x in filtro:
            pen = pendiente(x, len(fechas), listo_y[x], data_yml["target_production"]*percentage)
            b = indep(x, puntos(pen, x, b) + values_list[index.index(x)], pen)
            listo_y[x:] = list(map(lambda y: puntos(pen, y, b), range(x, len(fechas))))
        lista = listo_y
    df2 = pd.DataFrame({"ds": fechas, "y": lista})
    for x in range(0, 365):
        df2["y"][(df2["ds"].dt.month == names.loc[x]["month"]) & (df2["ds"].dt.day == names.loc[x]["day"])] = \
        df2["y"][(df2["ds"].dt.month == names.loc[x]["month"]) & (df2["ds"].dt.day == names.loc[x]["day"])] = \
        df2["y"][(df2["ds"].dt.month == names.loc[x]["month"]) & (df2["ds"].dt.day == names.loc[x]["day"])] + (
                    df2["y"][
                        (df2["ds"].dt.month == names.loc[x]["month"]) & (df2["ds"].dt.day == names.loc[x]["day"])] *
                    names.loc[x][f"{energy}_per"] - names.loc[x][f"Demand_per"])
    percentage_diff = diff_percentage(df2["y"].iloc[-1], data_yml["target_production"]*percentage)
    if percentage != 0.0:
        df2["y"] = df2["y"] + (df2["y"] * (percentage_diff / 100))
    else:
        df2["y"].iloc[-1] = 0
    #df2["y"] = df2["y"] +  (df2["y"] * (percentage_diff / 100))
    return df2,df_percentage


def file_exists(): #CHECK IF FILE EXISTS
    if os.path.isfile(f'output/{data_yml["pattern"]}'):
        return True
    else:
        print("File does not exist")
        return False

def diff_percentage(last_value,total): #DIFFERENCE PERCENTAGE
    return ((total-last_value)/last_value)*100


def regression_nuclear(df,percentage,energy): #NUCLEAR REGRESSION
    df[data_yml["date"]] = pd.to_datetime(df[data_yml["date"]],utc=True)
    fechas = pd.date_range(start=df[data_yml["date"]].max().replace(tzinfo=None), end=data_yml["objetive_date"], freq=data_yml["freq"])
    pen, b = math_.ecuación_de_la_recta(len(df), 0, len(df)+len(fechas), data_yml["target_production"]*percentage)
    lista = []
    offset = len(df)
    x = np.arange(offset, offset+len(fechas), 1)
    result = math_.calcular_puntos_de_la_recta(pen, b, len(df), x)
    lista.extend(result)
    df2 = pd.DataFrame({"ds": fechas, "y": lista})

    def pendiente(x1, x2, y1, y2):
        return ((y2 - y1) / (x2 - x1))

    def indep(x1, y1, m):
        return (y1 - (m * x1))

    def puntos(m, x, b):
        return ((m * x) + b)

    date_list, values_list = get_centrals(energy)
    index = []
    pivot = fechas.tolist()
    for d in date_list:
        if d in pivot:
            index.append(get_index(pivot, d))

    b = indep(0, 0, 0)
    listo_y = list(map(lambda x: puntos(0, x, b), range(0, len(fechas))))
    filtro = list(filter(lambda x: x in index, range(0, len(fechas))))
    for x in filtro:
        pen = 0
        b = indep(x, puntos(pen, x, b) + values_list[index.index(x)], pen)
        listo_y[x:] = list(map(lambda y: puntos(pen, y, b), range(x, len(fechas))))
    lista = listo_y
    return df2,lista

def main():
    import time
    start_time = time.time()
    if not file_exists():
        print("No existe el archivo de porcentajes nesesario para la predicción")
        print("Cerrando programa")
        exit()
    warnings.simplefilter('ignore') #IGNORE WARNINGS
    percentage_df = {}
    warnings.simplefilter(action='ignore', category=FutureWarning) #IGNORE WARNINGS
    df = pd.read_csv(data_yml["data"], parse_dates=[data_yml["date"]]) #READ CSV

    df[data_yml["generation"]] = df[data_yml["generation"]].astype(float) #CONVERT GENERATION TO FLOAT


    diccionario_renew = get_percentage_renew()
    total_dic = combine_dic(diccionario_renew, {})
    sum = 0
    for x in total_dic:
        sum = sum + (total_dic[x][0])

    if sum < 0.99 or sum > 1.0001:
        print("ERROR: The sum of the percentages is not 1")
        return
    else:
        print("The sum of the percentages is 1 all good")
    if data_yml["energy"]["Predict"] != None:#IF RENEWABLE IS IN TO PREDICT
        energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Predict"])))
        energys2 = (list(map(lambda x: x[0], energys)))  # CAN BE BETTER PLEASE CHANGE
        for renowable in energys2:
            if renowable == "Nuclear":
                result,nuclear_pivot = regression_nuclear(df, float(total_dic[renowable][0]),renowable)
                total_dic[renowable] = result["y"]
            else:
                df_renew = df[df[data_yml["tecnology"]] == renowable] # CONVERTIR EN FUNCION
                df2 = create_df(df_renew,renowable)
                df2 = df2[[data_yml["date"], data_yml["generation"]]]
                Serie = timeseries()
                Serie.name = renowable
                Serie.units = "MWh"
                Serie.time_units = "Date"
                Serie.data["y"] = df2[data_yml["generation"]]
                Serie.data["ds"] = df2[data_yml["date"]]
                result,percentages_df = Javier_Regression(df,Serie.convert_to_dataframe(), (float(total_dic[renowable][0])), renowable,percentage_df,get_type(renowable))
                total_dic[renowable] = result["y"]
        df_demand = pd.read_csv("data/Datos_demanda.csv", parse_dates=["fecha"])
        df_demand = df_demand.rename(columns={"fecha": "ds", "demanda": "y"})
        demand,void = Javier_Regression(Serie.convert_to_dataframe(),df_demand, 1, "Demand",percentage_df,"Demand")

    if not file_exists():
        percentage_df = pd.DataFrame(percentages_df)
        percentage_df.to_csv("Percentages.csv")

    if not os.path.exists("output"):
        os.makedirs("output")

    with pd.ExcelWriter('output/Forecast.xlsx') as writer: #SAVE TO EXCEL
        saving = pd.DataFrame(total_dic)
        #saving["Nuclear2"] = final_result
        if data_yml["groupby"] == "Day":
            saving.index = result["ds"].dt.date
        if data_yml["groupby"] == "Month":
            saving["date"] = result["ds"]
            saving = saving.groupby([saving["date"].dt.year,saving["date"].dt.month]).sum()
        if data_yml["groupby"] == "Year":
            saving["date"] = result["ds"]
            saving = saving.groupby(saving["date"].dt.year).sum()
        pivot = saving["Nuclear"]
        saving["Nuclear"] = nuclear_pivot
        saving["Total"] = saving.sum(axis=1)
        saving["Nuclear_Centrals_trend"] = pivot
        saving["Demand"] = list(demand["y"])
        saving.to_excel(writer, sheet_name='Forecast')
    end_time = time.time()
    print("Time: ", end_time - start_time)


if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/,
