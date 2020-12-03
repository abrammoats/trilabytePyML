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

def findMAPE(frame, options, seasonality):
    options = options.copy()
    options['seasonality'] = seasonality
    model = Forecast()
    return  model.forecast(frame.copy(), options)['MAPE']

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
            
            try:
                currentOptions = options.copy()
                
                model = Forecast()
                
                if (method == 'MLR'):
                    if params.getParam('seasonality', options) == 'Auto':
                        currentOptions['seasonality'] = findOptimalSeasonality(frame.copy(), options.copy())
                    
                    fdict = model.forecast(frame, currentOptions.copy())
                elif method.lower() == 'Prophet'.lower():
                    fdict = model.forecastProphet(frame, options.copy())
                else:
                    fdict = model.forecastARIMA(frame, currentOptions.copy())
                
                frame = fdict['frame']
                frame['X_ERROR'] = None 
                frame['X_METHOD'] = method
                
                outputFrame = frame if outputFrame is None else outputFrame.append(frame)
            
            except Exception as e:
                ed = str(traceback.format_exc()).replace('\n', ' ')
                frame['X_ERROR'] = ed
                frame['X_METHOD'] = method
                
                outputFrame = frame if outputFrame is None else outputFrame.append(frame)
    return outputFrame


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
    print("conda install -c anaconda ephem")
    print("conda install -c conda-forge pystan")
    print("conda install -c conda-forge fbprophet")
    print("-------------------------------")
    print("Usage: python -m trilabytePyML.AutoForecast [json forecast options] [csv source data] [output csv file]")
    print("-------------------------------")
  
    pd.options.mode.chained_assignment = None  # default='warn'
  
    fileName = 'c:/temp/retail_unit_demand3.csv'
    jsonFileName = 'c:/temp/retail_unit_demand_options.json'
    outputFileName = 'c:/temp/retail_unit_demand_forecast.csv'
    
#     if (len(sys.argv) < 3):
#         print("Error: Insufficient arguments")
#         sys.exit(-1)
#               
#     jsonFileName = sys.argv[1]
#     fileName = sys.argv[2]
#     outputFileName = sys.argv[3]
    
    with open(jsonFileName, 'r') as fp:
        options = json.load(fp)
    
    print('Options:') 
    print(json.dumps(options,indent=2), '\n')
    
    frame = pd.read_csv(fileName)
        
    outputFrame = splitFramesAndForecast(frame, options)
    
    outputFrame.to_csv(outputFileName, index=False)
    
    print("Output file: ", outputFileName)
    
    print("Forecast(s) complete...")
