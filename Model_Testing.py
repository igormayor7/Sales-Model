##1. Import Tools and Packages

#1.a Import General Tests
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy import stats

#1.b Import Statistical Tests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import acf

#1.c Import Statsmodels
from statsmodels.tsa.api import VAR

#1.d Import Dialogues
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import askyesno
from tkinter import messagebox as mb
from tkinter.filedialog import askopenfilename
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from prompt_toolkit import prompt

print('\n')

##2. Select Selling Out Data (CSV)

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


##3. Select Selling In Data (CSV)

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


##4. Select Parameters

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


##5. Read and Format Original Datasets

#5.a Selling Out
so = pd.read_csv(sofile)
so = so.dropna()
print("Selling Out Dataset:" + '\n')
print(so)
print('\n')

#5.b Selling In
si = pd.read_csv(sifile)
si = si.dropna()
print("Selling In Dataset:" + '\n')
print(si)
print('\n')

#5.c Format Selling Out
so = so.astype({"Selling Out":int})
so = so.astype({"Item Code":int})
so['Date']= pd.to_datetime (so['Date'], format = '%m/%d/%Y')
so = so.set_index('Date')
so = pd.DataFrame(so)

#5.d Slice Selling Out for selected Item Code
sos = so[so["Item Code"] == ic]
print("Selling Out Data for selected Item Code:" + '\n')
print(sos)
print('\n')

#5.e Selling Out BoxPlot 1 (Before removing outliers)
fig = plt.figure(figsize =(10, 7))
plt.boxplot(sos['Selling Out'])
plt.show()

#5.f Remove Selling Out outliers based on z score > +/-3
z = np.abs(stats.zscore(sos["Selling Out"]))
sos = sos[(z<3)]

#5.g Selling Out BoxPlot 2 (After removing outliers)
fig = plt.figure(figsize =(10, 7))
plt.boxplot(sos['Selling Out'])
plt.show()

#5.h Sum up Selling Out by month
sos = sos.groupby(pd.Grouper(freq='MS'))['Selling Out'].sum()
sos = pd.DataFrame(sos)
print("Monthly Selling Out Data for selected Item Code:" + '\n')
print(sos)
print('\n')

#5.i Format Selling In
si = si.astype({"Selling In":int})
si = si.astype({"Item Code":int})
si = si.astype({"Customer Code":int})
si['Date']= pd.to_datetime (si['Date'], format = '%Y %m %d')
si = si.set_index(["Date"])
si = pd.DataFrame(si)

#5.j Slice Selling In for selected Customer Code and Item Code
sis = si[si["Customer Code"] == cc]
sis = sis[sis["Item Code"] == ic]
print("Selling In Data for selected Customer and Item Code:" + '\n')
print(sis)
print('\n')

#5.k Selling In BoxPlot 1
fig = plt.figure(figsize =(10, 7))
plt.boxplot(sis['Selling In'])
plt.show()

#5.l Remove Selling In outliers based on z score > +/-3
z1 = np.abs(stats.zscore(sis["Selling In"]))
sis = sis[(z1<3)]

#5.m Selling In BoxPlot 2
fig = plt.figure(figsize =(10, 7))
plt.boxplot(sis['Selling In'])
plt.show()

#5.n Sum up Selling In by month
sis = sis.groupby(pd.Grouper(freq='MS'))['Selling In'].sum()
sis = pd.DataFrame(sis)
print("Monthly Selling In Data for selected Customer and Item Code:" + '\n')
print(sis)
print('\n')

#5.o Select the final month of the data (Trim longest to end at same point as shortest)
if sis.index[-1] > sos.index[-1]:
    x = sos.index[-1]
else:
    x = sis.index[-1]


#5.p Fill 0 for months with no Selling Out
d1 = pd.DataFrame(index=pd.date_range('2016-01-01',x,freq = 'MS'))
sos1 = sos.join(d1,how='right')
sos2 = sos1.fillna(0)
sos2.index.names = ["Date"]

#5.q Fill 0 for months with no Selling In
d2 = pd.DataFrame(index=pd.date_range('2016-01-01',x,freq = 'MS'))
sis1 = sis.join(d2,how='right')
sis2 = sis1.fillna(0)
sis2.index.names = ["Date"]


##6. Create final datatset

#6.a Merge Selling In and Selling Out Data
sales = pd.merge(sis2, sos2, left_index=True, right_index=True)
sales = sales.fillna(0)
print("Merged Selling In and Selling Out Data for selected Customer and Item Code: " + '\n')
print(sales)
print('\n')

#6.b Divide Selling Out Values by 100 for graphing
sales["Selling Out"] = (sales["Selling Out"]/100)
sales = sales.drop_duplicates(keep='last')
sales.index = pd.DatetimeIndex(sales.index)
sales.index.freq = "MS"

#6.c Print final dataset used for modelling
print("Full Merged Selling In and Selling Out Data for selected Customer and Item Code: " + '\n')
print(sales.to_string())
print('\n')
sales.plot()
plt.show()

#6.d Print descriptive statistics for the cleaned and merged dataset
print("Descriptive Statistics: " + '\n')
print(sales.describe())
print('\n')

##7. Perform Primary Statistical Tests

#7.a Perform Granger Causality Test at 0,05 significance level
maxlag=12
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


#7.b Perform Cointegration Test
def cointegration_test(sales, alpha=0.05):  
    out = coint_johansen(sales,-1,1)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    #Summary
    print('Cointegration   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(sales.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

print("Johanson's Cointegration Test: " + '\n')
print(cointegration_test(sales))


#7.c Perfrom ADFuller Test
print("\nAugmented Dickey-Fuller Test (ADFuller): " + '\n')
def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)
    #Print Summary
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
for name, column in sales[:-p].iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


#7.d Print ADFuller Test for each time series
result = adfuller(sales[:-p]["Selling Out"], autolag='AIC')
result1 = adfuller(sales[:-p]["Selling In"], autolag='AIC')
adf = float(f'{result[1]}')
adf1 = float(f'{result1[1]}')


##8. Define non-differenced VAR model as a function for future use
def varmodel():

    #8.a Set up VAR on the sales - p dataset
    var = VAR(sales[:-p])

    #8.b Select max lags for the model (We found that (total number of values)/3.5
    # is optimal for maxlag as 4 leads to a very small lag length)
    max_lags = sales[:-p]["Selling Out"].count()/3.5
    max_lags = math.ceil(max_lags)
    print("Max Lag Length: " + '\n')
    print(max_lags)
    print('\n')

    #8.c Produce AIC, BIC, FPE and HQIC fit scores for each lag up to maxlag
    fit_score = var.select_order(maxlags = max_lags)
    print("AIC, BIC, FPE and HAIC fit scores for each lag length up to Max Lag Length: " + '\n')
    print(fit_score.summary())
    print('\n')

    #8.d Model automatically fits the lag order with lowest AIC value
    model_fitted = var.fit(maxlags=max_lags, ic='aic')
    print("Fitted model summary: " + '\n')
    print(model_fitted.summary())

    #8.e Perform Durbin-Watson Test for Autocorrelation
    print("Durbin-Watson Test for Autocorrelation of residuals: " + '\n')
    from statsmodels.stats.stattools import durbin_watson
    out = durbin_watson(model_fitted.resid)
    for col, val in zip(sales[:-p].columns, out):
        print((col), ":", round(val, 2))
    print('\n')

    #8.f Set lag order to that selected previously
    lag_order = model_fitted.k_ar

    #8.g Forecast future values up to p periods
    results = model_fitted.forecast(sales[:-p].values[-lag_order:],p)
    model_fitted.plot_forecast(p)
    plt.show()

    #8.h Cut the original sales dataset to only have p last values 
    real = sales[-p:]

    #8.i Convert the forecasts into workable dataframes (tables)
    forecast = pd.DataFrame(results)
    forecast.columns = ['Selling In' , 'Selling Out']
    forecast.index = real.index.copy()
    print("Forecasted Selling In and Selling Out Values: " + '\n')
    print(forecast)
    print('\n')

    print("Descriptive Statistics of De-differenced forecast: " + '\n')
    print(forecast.describe())
    print('\n')

    #8.j Perform MAPE, ME, MAE, MPE, RMSE, corr and minmax accuracy tests for Selling In and Selling Out forecasts
    def forecast_accuracy(forecasted, actual):
        mape = np.mean(np.abs(forecasted - actual)/np.abs(actual))  # MAPE
        me = np.mean(forecasted - actual)             # ME
        mae = np.mean(np.abs(forecasted - actual))    # MAE
        mpe = np.mean((forecasted - actual)/actual)   # MPE
        rmse = np.mean((forecasted - actual)**2)**.5  # RMSE
        corr = np.corrcoef(forecasted, actual)[0,1]   # corr
        mins = np.amin(np.hstack([np.array(forecasted)[:,None], 
                                np.array(actual)[:,None]]), axis=1)
        maxs = np.amax(np.hstack([np.array(forecasted)[:,None], 
                                np.array(actual)[:,None]]), axis=1)
        minmax = 1 - np.mean(mins/maxs)             # minmax
        return({'mape':mape, 'me':me, 'mae': mae, 
                'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

    print("Forecast Accuracy Tests: " + '\n')

    print('Forecast Accuracy of: Selling In' + '\n')
    accuracy_prod = forecast_accuracy(forecast['Selling In'].values, real['Selling In'])
    for k, v in accuracy_prod.items():
        print(k, ': ', round(v,4))

    print('\nForecast Accuracy of: Selling Out' + '\n')
    accuracy_prod = forecast_accuracy(forecast['Selling Out'].values, real['Selling Out'])
    for k, v in accuracy_prod.items():
        print(k, ': ', round(v,4))
    print('\n')

    #8.k Convert all negative forecast values to 0
    forecast[forecast<0] = 0
    print("Forecasted Selling In and Selling Out Values with all negative values = 0: " + '\n')
    print(forecast)
    print('\n')

    #8.l Create one dataset with real and forecasted Sales In values
    sir = real["Selling In"]
    sif = forecast["Selling In"].clip(lower=0)
    sirf = [sir,sif]
    si = pd.concat(sirf, axis=1, join='inner')
    si.columns = ["Real" , "Forecasted"]
    print("Real vs Forecasted Selling In Values: " + '\n')
    print(si)
    print('\n')

    #8.m Plot Real vs Forecasted Data
    si = si.astype({"Real":int})
    si = si.astype({"Forecasted":int})
    si[['Real','Forecasted']].plot()
    plt.show()


##9. If one or more time series is non-stationary -> ask user whether they want to difference
if adf >= 0.05 or adf1 >= 0.05:
    
    ##10. Define function for user selection
    def call():
        global res
        global x
        res=mb.askyesno('Difference?','One or more of the time series is non-stationary, would you like to difference?')


        #10.a If user decided to difference -> difference dataset
        if res == True:
            root.destroy()
            sales1 = sales[:-p].diff(1)
            sales1 = sales1.dropna(thresh=2)
            print("Differenced Selling In and Selling Out Data: " + '\n')
            print(sales1)
            print('\n')

            #10.a.1 Plot differenced results
            sales1.plot()
            plt.show() 

            #10.a.2 Perform Granger Causality Test at 0,05 significance level
            maxlag=12
            test = 'ssr_chi2test'
            def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
                sales1 = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
                for c in sales1.columns:
                    for r in sales1.index:
                        test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                        p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                        if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                        min_p_value = np.min(p_values)
                        sales1.loc[r, c] = min_p_value
                sales1.columns = [var + '_x' for var in variables]
                sales1.index = [var + '_y' for var in variables]
                return sales1

            print("Grangers Causation Matrix at 0.05 significance level: " + '\n') 
            print(grangers_causation_matrix(sales1, variables = sales1.columns))
            print('\n')  

            #10.a.3 Perform Cointegration Test
            def cointegration_test(sales1, alpha=0.05):  
                out = coint_johansen(sales1,-1,1)
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

            #10.a.4 Perfrom ADFuller Test
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

            #10.a.5 Set up VAR on the differenced sales dataset
            var = VAR(sales1)

            #10.a.6 Select max lags for the model (We found that (total number of values)/3.5
            # is optimal for maxlag as 4 results in a very short lag length)
            max_lags = sales1["Selling Out"].count()/3.5
            max_lags = math.ceil(max_lags)
            print("Max Lag Length: " + '\n')
            print(max_lags)
            print('\n')

            #10.a.7 Produce AIC, BIC, FPE and HQIC fit scores for each lag up to maxlag
            fit_score = var.select_order(maxlags = max_lags)
            print("AIC, BIC, FPE and HAIC fit scores for each lag length up to Max Lag Length: " + '\n')
            print(fit_score.summary())
            print('\n')

            #10.a.8 Model automatically fits the lag order with lowest AIC value
            model_fitted = var.fit(maxlags = max_lags , ic='aic')
            print("Fitted Model Summary: " + '\n')
            print(model_fitted.summary())

            #10.a.9 Perform Durbin-Watson Test for Autocorrelation
            print("Durbin-Watson Test for Autocorrelation of residuals: " + '\n')
            from statsmodels.stats.stattools import durbin_watson
            out = durbin_watson(model_fitted.resid)
            for col, val in zip(sales1.columns, out):
                print((col), ":", round(val, 2))
            print('\n')

            #10.a.10 Set lag order to that selected previously
            lag_order = model_fitted.k_ar
            
            #10.a.11 Forecast future values up to p periods
            results = model_fitted.forecast(sales1.values[-lag_order:],p)
            model_fitted.plot_forecast(p)
            plt.show()

            #10.a.12 Cut the original sales dataset to only have p last values
            real = sales[-p:]

            #10.a.13 Convert the forecasts into workable dataframes (tables)
            forecast = pd.DataFrame(results)
            forecast.columns = ['Selling In' , 'Selling Out']
            forecast.index = real.index.copy()
            print("Forecasted Selling In and Selling Out Values from Differenced data: " + '\n')
            print(forecast)
            print('\n')

            #10.a.14 De-difference the dataset
            cols = sales.columns
            x = []
            for col in cols:
                diff_results = sales[col] + forecast[col].shift(-1)
                x.append(diff_results)
            forecastddf = pd.concat(x, axis=1)
            forecastddf = forecastddf.dropna()
            print("De-differenced forecasted Selling In and Selling Out Values: " + '\n')
            print(forecastddf)
            print('\n')
            
            print("Descriptive Statistics of De-differenced forecast: " + '\n')
            print(forecastddf.describe())
            print('\n')


            #10.a.15 Shorten the real values dataset as the forecasted values dataset is shortened when de-differencing
            real = real[:-1]

            #10.a.16 Perform MAPE, ME, MAE, MPE, RMSE, corr and minmax accuracy tests for Selling In and Selling Out forecasts
            def forecast_accuracy(forecasted, actual):
                mape = np.mean(np.abs(forecasted - actual)/np.abs(actual))  # MAPE
                me = np.mean(forecasted - actual)             # ME
                mae = np.mean(np.abs(forecasted - actual))    # MAE
                mpe = np.mean((forecasted - actual)/actual)   # MPE
                rmse = np.mean((forecasted - actual)**2)**.5  # RMSE
                corr = np.corrcoef(forecasted, actual)[0,1]   # corr
                mins = np.amin(np.hstack([np.array(forecasted)[:,None], 
                                        np.array(actual)[:,None]]), axis=1)
                maxs = np.amax(np.hstack([np.array(forecasted)[:,None], 
                                        np.array(actual)[:,None]]), axis=1)
                minmax = 1 - np.mean(mins/maxs)             # minmax
                return({'mape':mape, 'me':me, 'mae': mae, 
                        'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

            print("Forecast Accuracy Tests: " + '\n')

            print('Forecast Accuracy of: Selling In' + '\n')
            accuracy_prod = forecast_accuracy(forecastddf['Selling In'].values, real['Selling In'])
            for k, v in accuracy_prod.items():
                print(k, ': ', round(v,4))

            print('\nForecast Accuracy of: Selling Out' + '\n')
            accuracy_prod = forecast_accuracy(forecastddf['Selling Out'].values, real['Selling Out'])
            for k, v in accuracy_prod.items():
                print(k, ': ', round(v,4))
            print('\n')

            #10.a.17 Convert all negative forecast values to 0
            forecastddf[forecastddf<0] = 0
            print("Forecasted Selling In and Selling Out Values with all negative values = 0: " + '\n')
            print(forecastddf)
            print('\n')

            print("Descriptive Statistics of De-differenced forecast with all negative values = 0: " + '\n')
            print(forecastddf.describe())
            print('\n')

            #10.a.18 Create one dataset with real and forecasted Sales In values
            sir = real["Selling In"]
            sif = forecastddf["Selling In"].clip(lower=0)
            sirf = [sir,sif]
            si = pd.concat(sirf, axis=1, join='inner')
            si.columns = ["Real" , "Forecasted"]
            print("Real vs Forecasted Selling In Values: " + '\n')
            print(si)
            print('\n')

            #10.a.19 Plot Real vs Forecasted Data
            si = si.astype({"Real":int})
            si = si.astype({"Forecasted":int})
            si[['Real','Forecasted']].plot()
            plt.show()

    
        #10.b If user choses not to difference -> run VAR model function defined above
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


##If both series are stationary, run VAR model without differencing
else:
    varmodel()
