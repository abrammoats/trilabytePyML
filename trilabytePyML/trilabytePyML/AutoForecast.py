#
#    http://trilabyte.com
#    Trilabyte Python Machine Learning
#    Copyright (c) 2020 - Trilabyte
#    Author: Scott Mutchler
#    Contact: smutchler@trilabyte.com
#
# more comments

import json 
import sys
import pandas as pd 
from trilabytePyML.Forecast import Forecast
import trilabytePyML.util.Parameters as params
import traceback 
from statistics import median
from trilabytePyML.stats.Statistics import calcMAPE

def findMAPE(frame, options, seasonality):
    options = options.copy()
    options['seasonality'] = seasonality
    model = Forecast()
    return  model.forecastMLR(frame.copy(), options)['MAPE']


def findOptimalSeasonality(frame, options):
    nonNullRowCount = frame[params.getParam('targetColumn', options)].count()
    periodicity = params.getParam('periodicity', options)
    
    if nonNullRowCount < periodicity:
        return "None"
    
    noSeasonalityMAPE = findMAPE(frame, options.copy(), 'None')
    additiveMAPE = findMAPE(frame, options.copy(), 'Additive')
    multiplicativeMAPE = findMAPE(frame, options.copy(), 'Multiplicative')
    
    minMAPE = min(noSeasonalityMAPE, additiveMAPE, multiplicativeMAPE)
    
    # in case of ties this returns a hierarchy of simple to complex
    if (noSeasonalityMAPE == minMAPE):
        return "None"
    elif (additiveMAPE == minMAPE):
        return "Additive"
    else:
        return "Multiplicative"


def splitFramesAndForecast(frame, options):
    frame.sort_values(by=params.getParam('sortColumns', options), ascending=True, inplace=True)
    
    frames = list(frame.groupby(by=params.getParam('splitColumns', options)))
    
    outputFrame = None

    for frame in frames:
            frame = frame[1]
            frame.reset_index(drop=True, inplace=True)
            
            method = params.getParam('method', options)
            
            if method == 'Auto':
                opts = options.copy()
                opts['method'] = 'ARIMA'
                arimaFrame = forecastSingleFrame(frame.copy(), opts)
                arimaMAPE = 1E6 if 'X_MAPE' not in arimaFrame else arimaFrame['X_MAPE'][0]
                
                opts = options.copy()
                opts['method'] = 'Prophet'
                prophetFrame = forecastSingleFrame(frame.copy(), opts)
                prophetMAPE = 1E6 if 'X_MAPE' not in prophetFrame else prophetFrame['X_MAPE'][0]
                
                opts = options.copy()
                opts['method'] = 'MLR'
                mlrFrame = forecastSingleFrame(frame.copy(), opts)
                mlrMAPE = 1E6 if 'X_MAPE' not in mlrFrame else mlrFrame['X_MAPE'][0]
                
                if 'X_FORECAST' in mlrFrame and 'X_FORECAST' in prophetFrame and 'X_FORECAST' in arimaFrame:
                    ensembleFrame = mlrFrame.copy() 
                    ensembleFrame['X_FORECAST'] = list(map(lambda x, y , z: median([x, y, z]), mlrFrame['X_FORECAST'], arimaFrame['X_FORECAST'], prophetFrame['X_FORECAST']))
                    ensembleMAPE = calcMAPE(ensembleFrame['X_FORECAST'], ensembleFrame[params.getParam('targetColumn', options)])
                    ensembleFrame['X_MAPE'] = ensembleMAPE
                    
                    mapes = [mlrMAPE, arimaMAPE, prophetMAPE, ensembleMAPE]
                else:
                    mapes = [mlrMAPE, arimaMAPE, prophetMAPE]
                
                print("Auto MAPEs (MLR, ARIMA, Prophet, Ensemble): ", mapes)
                
                minMAPE = min(mapes)
                
                if (mlrMAPE <= minMAPE):
                    frame = mlrFrame
                elif (prophetMAPE <= minMAPE):
                    frame = prophetFrame
                elif (arimaMAPE <= minMAPE):
                    frame = arimaFrame      
                else:
                    frame = ensembleFrame                         
                
            else:
                frame = forecastSingleFrame(frame, options.copy())
            
            outputFrame = frame if outputFrame is None else outputFrame.append(frame, ignore_index=True)
    
    return outputFrame

def forecastSingleFrame(frame, options):
    try:
        method = params.getParam('method', options)
        currentOptions = options.copy()
        
        model = Forecast()
                    
        if (method == 'MLR'):
            if params.getParam('seasonality', options) == 'Auto':
                currentOptions['seasonality'] = findOptimalSeasonality(frame.copy(), options.copy())
            
            fdict = model.forecastMLR(frame, currentOptions.copy())
        elif method.lower() == 'Prophet'.lower():
            fdict = model.forecastProphet(frame, options.copy())
        else:
            fdict = model.forecastARIMA(frame, currentOptions.copy())
        
        frame = fdict['frame']
        frame['X_ERROR'] = None 
        frame['X_METHOD'] = method
    
    except Exception as e:
        ed = str(traceback.format_exc()).replace('\n', ' ')
        frame['X_ERROR'] = ed
        frame['X_METHOD'] = method
    
    return frame
        

##############################
# Main
##############################
if __name__ == '__main__':
    
    print("AutoForecast")
    print("-------------------------------")
    print("*** You must use Anaconda for Facebook Prophet")
    print("")
    print("Required Librarires:")
    print("pip install pandas loess scipy numpy scikit-learn pmdarima")
    print("conda install -c anaconda ephem pystan fbprophet")
    print("-------------------------------")
    print("Usage: python -m trilabytePyML.AutoForecast [json forecastMLR options] [csv source data] [output csv file]")
    print("-------------------------------")
  
    pd.options.mode.chained_assignment = None  # default='warn'
  
    DEBUG = True 
  
    if DEBUG:
        fileName = 'c:/temp/retail_unit_demand.csv'
        jsonFileName = 'c:/temp/retail_unit_demand_options.json'
        outputFileName = 'c:/temp/retail_unit_demand_forecast.csv'
    else:
        if (len(sys.argv) < 3):
            print("Error: Insufficient arguments")
            sys.exit(-1)
                    
        jsonFileName = sys.argv[1]
        fileName = sys.argv[2]
        outputFileName = sys.argv[3]
    
    with open(jsonFileName, 'r') as fp:
        options = json.load(fp)
    
    print('Options:') 
    print(json.dumps(options, indent=2), '\n')
    
    frame = pd.read_csv(fileName)
        
    outputFrame = splitFramesAndForecast(frame, options)
    
    outputFrame.to_csv(outputFileName, index=False)
    
    print("Output file: ", outputFileName)
    
    print("Forecast(s) complete...")
