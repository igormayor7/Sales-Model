##Import Tools and Packages
import math
from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
from dateutil.relativedelta import relativedelta

#Import Statistical Tests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import acf

#Import Statsmodels
from statsmodels.tsa.api import VAR

#Import Dialogues
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import askyesno
from tkinter import messagebox as mb
from tkinter.filedialog import askopenfilename
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from prompt_toolkit import prompt

print('\n')

##Select Selling Out Data (CSV)

rootso = tk.Tk()
rootso.title('Tkinter Open File Dialog')
rootso.resizable(False, False)
rootso.geometry('300x150')

def select_filesi():
    global sofile
    sofile = fd.askopenfilename(title='Open a file',initialdir='/')
    showinfo(title='Selected File',message=sofile)
    rootso.destroy()

open_button = ttk.Button(rootso,text='Select Selling Out Data',command=select_filesi)
open_button.pack(expand=True)

rootso.mainloop()
print("Selected Selling Out dataset: " + sofile + '\n')


##Select Selling In Data (CSV)

rootsi = tk.Tk()
rootsi.title('Tkinter Open File Dialog')
rootsi.resizable(False, False)
rootsi.geometry('300x150')

def select_fileso():
    global sifile
    sifile = fd.askopenfilename(title='Open a file',initialdir='/')
    showinfo(title='Selected File',message=sifile)
    rootsi.destroy()

open_button = ttk.Button(rootsi,text='Select Selling In Data',command=select_fileso)
open_button.pack(expand=True)

rootsi.mainloop()
print("Selected Selling In dataset: " + sifile + '\n')


##Select Parameters

def run():
    global cc
    global ic
    global p

    cc = int(cce.get())
    ic = int(ice.get())
    p = int(pe.get())
    master.destroy()

master = tk.Tk()
tk.Label(master, text="Customer Code").grid(row=0)
tk.Label(master, text="Item Code").grid(row=1)
tk.Label(master, text="Forecast Period in Months").grid(row=2)

cce = tk.Entry(master)
ice = tk.Entry(master)
pe = tk.Entry(master)

cce.grid(row=0, column=1)
ice.grid(row=1, column=1)
pe.grid(row=2, column=1)

tk.Button(master, text='Quit', command=master.quit).grid(row=5, column=0, sticky=tk.W, pady=4)
tk.Button(master, text='Run Forecast', command=run).grid(row=5, column=1, sticky=tk.W, pady=4)

tk.mainloop()

print("Selected Customer Code: " + str(cc) + '\n')
print("Selected Item Code: " + str(ic) +'\n')
print("Selected Forecast Period: " + str(p) + '\n')


##Read and Format Original Datasets

#Selling Out
so = pd.read_csv(sofile)
so = so.dropna()
print("Selling Out Dataset:" + '\n')
print(so)
print('\n')

#Selling In
si = pd.read_csv(sifile)
si = si.dropna()
print("Selling In Dataset:" + '\n')
print(si)
print('\n')

#Format Selling Out
so = so.astype({"Selling Out":int})
so = so.astype({"Item Code":int})
so['Date']= pd.to_datetime (so['Date'], format = '%m/%d/%Y')
so = so.set_index('Date')
so = pd.DataFrame(so)

#Slice Selling Out for selected Item Code
sos = so[so["Item Code"] == ic]
print("Selling Out Data for selected Item Code:" + '\n')
print(sos)
print('\n')

#Selling Out BoxPlot 1 (Before removing outliers)
fig = plt.figure(figsize =(10, 7))
plt.boxplot(sos['Selling Out'])
plt.show()

#Remove Selling Out outliers based on z score > +/-3
z = np.abs(stats.zscore(sos["Selling Out"]))
sos = sos[(z<3)]

#Selling Out BoxPlot 2 (After removing outliers)
fig = plt.figure(figsize =(10, 7))
plt.boxplot(sos['Selling Out'])
plt.show()

#Sum up Selling Out by month
sos = sos.groupby(pd.Grouper(freq='MS'))['Selling Out'].sum()
sos = pd.DataFrame(sos)
print("Monthly Selling Out Data for selected Item Code:" + '\n')
print(sos)
print('\n')

#Format Selling In
si = si.astype({"Selling In":int})
si = si.astype({"Item Code":int})
si = si.astype({"Customer Code":int})
si['Date']= pd.to_datetime (si['Date'], format = '%Y %m %d')
si = si.set_index(["Date"])
si = pd.DataFrame(si)

#Slice Selling In for selected Customer Code and Item Code
sis = si[si["Customer Code"] == cc]
sis = sis[sis["Item Code"] == ic]
print("Selling In Data for selected Customer and Item Code:" + '\n')
print(sis)
print('\n')

#Selling In BoxPlot 1
fig = plt.figure(figsize =(10, 7))
plt.boxplot(sis['Selling In'])
plt.show()

#Remove Selling In outliers based on z score > +/-3
z1 = np.abs(stats.zscore(sis["Selling In"]))
sis = sis[(z1<3)]

#Selling In BoxPlot 2
fig = plt.figure(figsize =(10, 7))
plt.boxplot(sis['Selling In'])
plt.show()

#Sum up Selling In by month
sis = sis.groupby(pd.Grouper(freq='MS'))['Selling In'].sum()
sis = pd.DataFrame(sis)
print("Monthly Selling In Data for selected Customer and Item Code:" + '\n')
print(sis)
print('\n')

#Select the final month of the data (Trim longest to end at same point as shortest)
if sis.index[-1] > sos.index[-1]:
    x = sos.index[-1]
else:
    x = sis.index[-1]


#Fill 0 for months with no Selling Out
d1 = pd.DataFrame(index=pd.date_range('2016-01-01',x,freq = 'MS'))
sos1 = sos.join(d1,how='right')
sos2 = sos1.fillna(0)
sos2.index.names = ["Date"]

#Fill 0 for months with no Selling In
d2 = pd.DataFrame(index=pd.date_range('2016-01-01',x,freq = 'MS'))
sis1 = sis.join(d2,how='right')
sis2 = sis1.fillna(0)
sis2.index.names = ["Date"]


##Create final datatset

#Merge Selling In and Selling Out Data
sales = pd.merge(sis2, sos2, left_index=True, right_index=True)
sales = sales.fillna(0)
print("Merged Selling In and Selling Out Data for selected Customer and Item Code: " + '\n')
print(sales)
print('\n')

#Divide Selling Out Values by 100 for graphing
sales["Selling Out"] = (sales["Selling Out"]/100)
sales = sales.drop_duplicates()
sales.index.freq = "MS"

#Print final dataset used for modelling
print("Full Merged Selling In and Selling Out Data for selected Customer and Item Code: " + '\n')
print(sales.to_string())
print('\n')
sales.plot()
plt.show()

#Print descriptive statistics for the cleaned and merged dataset
print("Descriptive Statistics: " + '\n')
print(sales.describe())
print('\n')

##Set up model

#Perform Granger Causality Test at 0,05 significance level
maxlag=20
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    sales = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in sales.columns:
        for r in sales.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            sales.loc[r, c] = min_p_value
    sales.columns = [var + '_x' for var in variables]
    sales.index = [var + '_y' for var in variables]
    return sales

print("Grangers Causation Matrix at 0.05 significance level: " + '\n') 
print(grangers_causation_matrix(sales, variables = sales.columns))
print('\n')  


#Perform Cointegration Test
def cointegration_test(sales, alpha=0.05):  
    out = coint_johansen(sales,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Cointegration   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(sales.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

print("Johanson's Cointegration Test: " + '\n')
print(cointegration_test(sales))


#Perfrom ADFuller Test
print("\nAugmented Dickey-Fuller Test (ADFuller): " + '\n')
def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)
    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')
    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')
    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")
for name, column in sales.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


#Print ADFuller Test for each time series
result = adfuller(sales["Selling Out"], autolag='AIC')
result1 = adfuller(sales["Selling In"], autolag='AIC')
adf = float(f'{result[1]}')
adf1 = float(f'{result1[1]}')


##Define non-differenced VAR model as a function for future use
def varmodel():
    
    global fname

    #Set up VAR on the sales dataset
    var = VAR(sales)

    #Select max lags for the model (We found that (total number of values)/3.5
    # is optimal for maxlag as any more usually produces an error)
    max_lags = sales["Selling Out"].count()/3.5
    max_lags = math.ceil(max_lags)
    print("Max Lag Length: " + '\n')
    print(max_lags)
    print('\n')

    #Produce AIC, BIC, FPE and HQIC fit measurements for each lag up to maxlag
    fit_score = var.select_order(maxlags = max_lags)
    print("AIC, BIC, FPE and HAIC fit scores for each lag length up to Max Lag Length: " + '\n')
    print(fit_score.summary())
    print('\n')

    #Model automatically fits the lag order with lowest AIC value
    model_fitted = var.fit(maxlags=max_lags, ic='aic')
    print("Fitted model summary: " + '\n')
    print(model_fitted.summary())

    #Perform Durbin-Watson Test for Autocorrelation
    print("Durbin-Watson Test for Autocorrelation of residuals: " + '\n')
    from statsmodels.stats.stattools import durbin_watson
    out = durbin_watson(model_fitted.resid)
    for col, val in zip(sales.columns, out):
        print((col), ":", round(val, 2))
    print('\n')

    #Set lag order to that selected previously
    lag_order = model_fitted.k_ar

    #Forecast future values up to p periods
    results = model_fitted.forecast(sales.values[-lag_order:],p)
    model_fitted.plot_forecast(p)
    plt.show()
    

    #Convert the forecasts into workable dataframes (tables)
    forecast = pd.DataFrame(results)
    forecast.columns = ['Selling In' , 'Selling Out']
    print(forecast)

    #Set index of forecasted values = last index value of sales dataset + 1 -> last index value of dataset + 1 + p
    y = sales.index[-1]
    y = pd.to_datetime(y)
    x = y + relativedelta(months = p)
    d3 = pd.DataFrame(index=pd.date_range(y, x,freq = 'MS'))
    d3 = d3[1:]
    forecast = forecast.set_index(d3.index)

    #Format forecasted dataset
    forecast = forecast.fillna(0)
    forecast.index.names = ["Date"]
    forecast.columns = ['Selling In' , 'Selling Out']
    forecast = pd.DataFrame(forecast)
    print("Forecasted Selling In and Selling Out Values from Differenced data: " + '\n')
    print(forecast)
    print('\n')

    #Join differenced sales and forecasted values datasets
    forecast1 = sales.append(forecast, ignore_index=False)
    print("Merged Real and Forecasted Selling In and Selling Out Values" + '\n')
    print(forecast1)
    print('\n')
    forecast1.plot()
    plt.show()

    #Convert all negative forecast values to 0
    forecast1[forecast1<0] = 0
    print("Forecasted Selling In and Selling Out Values with all negative values = 0: " + '\n')
    print(forecast1.to_string())
    print('\n')
    forecast1.plot()
    plt.show()

    def call():
        global res
        global x
        global fname
        res=mb.askyesno('Save?','Would you like to save the forecasts?')

        if res == True:
            root1.destroy()

            def run():
                global fname
                fname = fnamee.get()
                master1.destroy()

            master1 = tk.Tk()
            tk.Label(master1, text="File Name").grid(row=0)

            fnamee = tk.Entry(master1)

            fnamee.grid(row=0, column=1)

            tk.Button(master1, text='Quit', command=master1.quit).grid(row=5, column=1, sticky=tk.W, pady=4)
            tk.Button(master1, text='Save', command=run).grid(row=5, column=2, sticky=tk.W, pady=4)

            tk.mainloop()

        else:   
            print("Forecast Not Saved" + '\n')

    #Criteria and launch for the saving request
    root1=tk.Tk()
    canvas=tk.Canvas(root1, width=200, height=200)
    canvas.pack()
    b=tk.Button(root1, text='Save?', command=call)
    canvas.create_window(100, 100, window=b)
    root1.mainloop()

    #Save forecasts if user choses to
    print("Selected File Name: " + str(fname) + '\n')
    fname = fname + ".xlsx"
    forecast1.to_excel(fname)
    print("DataFrame written to Excel successfully" + '\n')


#If one or more time series is non-stationary -> ask user whether they want to difference
if adf >= 0.05 or adf1 >= 0.05:
    
    #Define function for user selection
    def call():
        global res
        global x
        res=mb.askyesno('Difference?','One or more of the time series is non-stationary, would you like to difference?')


        #If user decided to difference -> difference dataset
        if res == True:
            root.destroy()
            sales1 = sales.diff(1)
            sales1 = sales1.dropna(thresh=2)
            print("Differenced Selling In and Selling Out Data: " + '\n')
            print(sales1)
            print('\n')

            #Plot differenced results
            sales1.plot()
            plt.show() 

            #Perform Cointegration Test
            def cointegration_test(sales1, alpha=0.05):  
                out = coint_johansen(sales1,-1,5)
                d = {'0.90':0, '0.95':1, '0.99':2}
                traces = out.lr1
                cvts = out.cvt[:, d[str(1-alpha)]]
                def adjust(val, length= 6): return str(val).ljust(length)

                # Summary
                print('Cointegration   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
                for col, trace, cvt in zip(sales1.columns, traces, cvts):
                    print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

            print("Johanson's Cointegration Test: " + '\n')
            print(cointegration_test(sales1))

            #Perfrom ADFuller Test
            print("\nAugmented Dickey-Fuller Test (ADFuller): " + '\n')
            def adfuller_test(series, signif=0.05, name='', verbose=False):
                r = adfuller(series, autolag='AIC')
                output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
                p_value = output['pvalue'] 
                def adjust(val, length= 6): return str(val).ljust(length)
                # Print Summary
                print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
                print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
                print(f' Significance Level    = {signif}')
                print(f' Test Statistic        = {output["test_statistic"]}')
                print(f' No. Lags Chosen       = {output["n_lags"]}')
                for key,val in r[4].items():
                    print(f' Critical value {adjust(key)} = {round(val, 3)}')
                if p_value <= signif:
                    print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
                    print(f" => Series is Stationary.")
                else:
                    print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
                    print(f" => Series is Non-Stationary.")
            for name, column in sales1.iteritems():
                adfuller_test(column, name=column.name)
                print('\n')

            #Set up VAR on the differenced sales dataset
            var = VAR(sales1)

            #Select max lags for the model (We found that (total number of values)/3.5
            # is optimal for maxlag as any more usually produces an error)
            max_lags = sales1["Selling Out"].count()/3.5
            max_lags = math.ceil(max_lags)
            print("Max Lag Length: " + '\n')
            print(max_lags)
            print('\n')

            #Produce fit scores in a table
            fit_score = var.select_order(maxlags = max_lags)
            print("AIC, BIC, FPE and HAIC fit scores for each lag length up to Max Lag Length: " + '\n')
            print(fit_score.summary())
            print('\n')

            #Model automatically fits the lag order with lowest AIC value
            model_fitted = var.fit(maxlags = max_lags , ic='aic')
            print("Fitted Model Summary: " + '\n')
            print(model_fitted.summary())

            #Perform Durbin-Watson Test for Autocorrelation
            print("Durbin-Watson Test for Autocorrelation of residuals: " + '\n')
            from statsmodels.stats.stattools import durbin_watson
            out = durbin_watson(model_fitted.resid)
            for col, val in zip(sales1.columns, out):
                print((col), ":", round(val, 2))
            print('\n')

            #Set lag order to that selected previously
            lag_order = model_fitted.k_ar
            
            #Forecast future values up to p periods
            results = model_fitted.forecast(sales1.values[-lag_order:],p)
            model_fitted.plot_forecast(p)
            plt.show()

            #Convert the forecasts into workable dataframes (tables)
            forecast = pd.DataFrame(results)
            forecast.columns = ['Selling In' , 'Selling Out']
            print(forecast)

            #Set index of forecasted values = last index value of sales dataset + 1 -> last index value of dataset + 1 + p
            y = sales.index[-1]
            y = pd.to_datetime(y)
            x = y + relativedelta(months = p)
            d3 = pd.DataFrame(index=pd.date_range(y, x,freq = 'MS'))
            d3 = d3[1:]
            forecast = forecast.set_index(d3.index)

            #Format forecasted dataset
            forecast = forecast.fillna(0)
            forecast.index.names = ["Date"]
            forecast.columns = ['Selling In' , 'Selling Out']
            forecast = pd.DataFrame(forecast)
            print("Forecasted Selling In and Selling Out Values from Differenced data: " + '\n')
            print(forecast)
            print('\n')

            #Join differenced sales and forecasted values datasets
            forecast1 = sales1.append(forecast, ignore_index=False)
            print(forecast1.to_string())

            #Define de-differencing function
            def invert_transformation(sales, forecast1, second_diff=False):
                """Revert back the differencing to get the forecast to original scale."""
                df_fc = forecast1.copy()
                columns = sales.columns
                for col in columns:        
                    df_fc[str(col)] = sales[col].iloc[-1] + df_fc[str(col)].cumsum()
                return df_fc

            #De-difference Forecasted Selling In and Selling Out Values
            forecastddf = invert_transformation(sales, forecast1, second_diff=True)  
            print(forecastddf)      
            print("De-differenced forecasted Selling In and Selling Out Values: " + '\n')
            print(forecastddf.to_string())
            print('\n')
            forecastddf.plot()
            plt.show()

            #Convert all negative forecast values to 0
            forecastddf[forecastddf<0] = 0
            print("Forecasted Selling In and Selling Out Values with all negative values = 0: " + '\n')
            print(forecastddf)
            print('\n')
            forecastddf.plot()
            plt.show()

            def call():
                global res
                global x
                global fname
                res=mb.askyesno('Save?','Would you like to save the forecasts?')

                if res == True:
                    root1.destroy()

                    def run():
                        global fname
                        fname = fnamee.get()
                        master1.destroy()

                    master1 = tk.Tk()
                    tk.Label(master1, text="File Name").grid(row=0)

                    fnamee = tk.Entry(master1)

                    fnamee.grid(row=0, column=1)

                    tk.Button(master1, text='Quit', command=master1.quit).grid(row=5, column=1, sticky=tk.W, pady=4)
                    tk.Button(master1, text='Save', command=run).grid(row=5, column=2, sticky=tk.W, pady=4)

                    tk.mainloop()
                    

                else:   
                    print("Forecast Not Saved" + '\n')
            
            #Criteria and launch for the saving request
            root1=tk.Tk()
            canvas=tk.Canvas(root1, width=200, height=200)
            canvas.pack()
            b=tk.Button(root1, text='Save?', command=call)
            canvas.create_window(100, 100, window=b)
            root1.mainloop()

            #Save forecasts if user choses to
            print("Selected File Name: " + str(fname) + '\n')
            forecastddf.to_excel(fname)
            print("DataFrame written to Excel successfully" + '\n')

        #If user choses not to difference -> run VAR model function defined above
        elif res == False:
            root.destroy()
            varmodel()


    #Criteria and launch for the differencing request
    root=tk.Tk()
    canvas=tk.Canvas(root, width=200, height=200)
    canvas.pack()
    b=tk.Button(root, text='Difference?', command=call)
    canvas.create_window(100, 100, window=b)
    root.mainloop()


#If both series are stationary, run VAR model without differencing
else:
    varmodel()
