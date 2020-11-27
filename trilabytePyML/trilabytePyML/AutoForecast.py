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
import traceback 

def buildSampleoptionsJSONFile(jsonFileName):
    options = dict()
    options['sortColumns'] = ['SKU', 'STORE_NUMBER', 'CAL_YEAR', 'CAL_MONTH']
    options['splitColumns'] = ['SKU', 'STORE_NUMBER']
    options['predictorColumns'] = ['X_INDEX', 'PREDICTOR_1', 'PREDICTOR_2', 'PREDICTOR_3', 'PREDICTOR_4']
    options['targetColumn'] = 'UNIT_DEMAND' 
    options['periodicity'] = 12
    options['seasonality'] = 'Multiplicative' #'Auto'  # 'Auto','None','Additive','Multiplicative' 

    options['method'] = 'ARIMA' #'ARIMA','MLR'
  
    options['autoDetectOutliers'] = True
    options['outlierStdevMultiplier'] = 3.0
   
    options['seasonalityBandwidth'] = 0.6
    options['ridgeAlpha'] = 1.0
    options['adjustSeasonalityBasedOnTrend'] = True 
    options['adjustSeasonalityTrendColumn'] = 'X_TREND' 
    
    options['forceNonNegative'] = False
    
    with open(jsonFileName, 'w') as fp:
        json.dump(options, fp)

def findMAPE(frame, options, seasonality):
    options = options.copy()
    options['seasonality'] = seasonality
    model = Forecast()
    return  model.forecast(frame.copy(), options)['MAPE']

def findOptimalSeasonality(frame, options):
    nonNullRowCount = frame[options['targetColumn']].count()
    periodicity = options['periodicity']
    
#     print("periodicity:", periodicity, "nonNullRowCount:", nonNullRowCount)
    
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
    frame.sort_values(by=options['sortColumns'], ascending=True, inplace=True)
    
    frames = list(frame.groupby(by=options['splitColumns']))
    
    outputFrame = None

    for frame in frames:
            frame = frame[1]
            frame.reset_index(drop=True, inplace=True)
            
            try:
                currentOptions = options.copy()
                
                model = Forecast()
                method = 'MLR' if not('method' in options) else options['method']
                
                if (method == 'MLR'):
                    if options['seasonality'] == 'Auto':
                        currentOptions['seasonality'] = findOptimalSeasonality(frame.copy(), options.copy())
                    
                    fdict = model.forecast(frame, currentOptions.copy())
                else:
                    fdict = model.forecastARIMA(frame, currentOptions.copy())
                
                frame = fdict['frame']
                frame['X_ERROR'] = None 
                
                outputFrame = frame if outputFrame is None else outputFrame.append(frame)
            
            except Exception as e:
                ed = traceback.format_exc().splitlines()
                ed = ed[-3] + ' ' + ed[-2] + ' ' + ed[-1]
                frame['X_ERROR'] = ed
    
                outputFrame = frame if outputFrame is None else outputFrame.append(frame)
    return outputFrame


##############################
# Main
##############################
if __name__ == '__main__':
    
    print("AutoForecast")
    print("-------------------------------")
    print("Required Librarires:")
    print("pip install pandas loess scipy numpy scikit-learn")
    print("-------------------------------")
    print("Usage: python -m trilabytePyML.AutoForecast [json forecast options] [csv source data] [output csv file]")
    print("-------------------------------")
  
    pd.options.mode.chained_assignment = None  # default='warn'
  
#     fileName = 'c:/temp/retail_unit_demand.csv'
#     jsonFileName = 'c:/temp/retail_unit_demand_options.json'
#     outputFileName = 'c:/temp/retail_unit_demand_forecast.csv'
    
    if (len(sys.argv) < 3):
        print("Error: Insufficient arguments")
        sys.exit(-1)
            
    jsonFileName = sys.argv[1]
    fileName = sys.argv[2]
    outputFileName = sys.argv[3]
    
    with open(jsonFileName, 'r') as fp:
        options = json.load(fp)
    
    print('Options:') 
    print(json.dumps(options,indent=2), '\n')
    
    frame = pd.read_csv(fileName)
        
    outputFrame = splitFramesAndForecast(frame, options)
    
    outputFrame.to_csv(outputFileName, index=False)
    
    print("Output file: ", outputFileName)
    
    print("Forecast(s) complete...")
