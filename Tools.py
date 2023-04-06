
import warnings

import pandas as pd
import sys

import math_

import matplotlib.pyplot as plt
import logging
import time
import yaml
from yaml.loader import SafeLoader
from Model import Model
import xlsxwriter
from datetime import datetime

from MPTE import timeseries

try: #LOAD CONFIG FILE
    with open(sys.argv[1],encoding="utf-8") as f:
        data_yml = yaml.load(f, Loader=SafeLoader)
        print(data_yml["archive"])
except: #IF NO FILE IS PASSED AS ARGUMENT
    with open("Chile.yml",encoding="utf-8") as f: #LOAD DEFAULT CONFIG FILE
        data_yml = yaml.load(f, Loader=SafeLoader)
        print(data_yml["archive"])

def save_excel(df,name,writer): #SAVE TIMESERIES TO EXCEL
    try:
        df.to_excel(writer, sheet_name=name)
        logging.info(f"{time.ctime(int(time.time()))} File saved successfully")
    except Exception as e:
        print("Error saving file")
        print(e)
        logging.error(f"{time.ctime(int(time.time()))} Error saving file in {name}")

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
    print(energys2)
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
    print(energys2)
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

def predict_future_logitic(df,periods,cap): #PREDICT FUTURE VALUES
    model = Model()
    df["cap"] = cap
    df.columns = ["y", "ds", "cap"]
   # print(df)
    model.fit(df) #FIT MODEL
    future = model.future(periods,cap) #
   # print(future)
    forecast = model.predict(future) #PREDICT FUTURE VALUES
    model.model_plot(forecast) #PLOT PREDICTION

    return forecast

def get_percentage_renew():
    energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Renewable"]))) # CAN BE BETTER PLEASE CHANGE
    energys2 = (list(map(lambda x: x[0], energys))) # CAN BE BETTER PLEASE CHANGE
    perce = (list(map(lambda x: list(x.values()), data_yml["energy"]["Renewable"]))) # CAN BE BETTER PLEASE CHANGE
    perce2 = (list(map(lambda x: x[0], perce))) #PERCENTAGE # CAN BE BETTER PLEASE CHANGE
    diccionario = dict(zip(energys2, perce2))
    return diccionario

def get_percentage_fossil():
    energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Fossil"]))) # CAN BE BETTER PLEASE CHANGE
    energys2 = (list(map(lambda x: x[0], energys))) # CAN BE BETTER PLEASE CHANGE
    perce = (list(map(lambda x: list(x.values()), data_yml["energy"]["Fossil"]))) # CAN BE BETTER PLEASE CHANGE
    perce2 = (list(map(lambda x: x[0], perce))) #PERCENTAGE # CAN BE BETTER PLEASE CHANGE
    diccionario = dict(zip(energys2, perce2))
    return diccionario

def combine_dic(dic1,dic2,dic3={},dic4={}): #COMBINE DICTIONARIES
    dic = {**dic1, **dic2, **dic3, **dic4}
    return dic

def make_cap_list(max_inicio,Total_predict,x_max): #MAKE CAP LIST
    m,b = mathematics.ecuación_de_la_recta(max_inicio, 0, x_max, Total_predict)
    lista = []
    for i in range(0,x_max):
        y_2 = mathematics.calcular_puntos_de_la_recta(m, b, i,i)
        lista.append(y_2)
    return list

def diff_bewteen_dates(date1,date2): #DIFF BETWEEN DATES
    date1 = datetime.strptime(date1, '%d/%m/%Y')
    diff = date2 - date1
    return diff.days

def make_date_from_objetive():
    objetive= (datetime.strptime(data_yml["objetive_date"], '%d/%m/%Y'))
    return objetive
