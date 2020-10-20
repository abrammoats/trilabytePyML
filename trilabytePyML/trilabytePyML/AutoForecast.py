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


def buildSampleoptionsJSONFile(jsonFileName):
    options = dict()
    options['sortColumns'] = ['SKU', 'STORE_NUMBER', 'CAL_YEAR', 'CAL_MONTH']
    options['splitColumns'] = ['SKU', 'STORE_NUMBER']
    options['predictorColumns'] = ['X_INDEX', 'PREDICTOR_1', 'PREDICTOR_2', 'PREDICTOR_3', 'PREDICTOR_4']
    options['targetColumn'] = 'UNIT_DEMAND' 
    options['periodicity'] = 12
    options['seasonality'] = 'Multiplicative' #'Auto'  # 'Auto','None','Additive','Multiplicative' 
  
    # options['outlierColumn'] = 'OUTLIER'
    options['autoDetectOutliers'] = True
   
    options['seasonalityBandwidth'] = 0.6
    options['ridgeAlpha'] = 1.0
    options['minMAPEImprovmentForOutlier'] = 5.0
    options['adjustSeasonalityBasedOnTrend'] = True 
    options['adjustSeasonalityTrendColumn'] = 'X_TREND' 
    
    # print(json.dumps(options))
    
    with open(jsonFileName, 'w') as fp:
        json.dump(options, fp)


def detectOutliers(frame, options):
    rowCount = frame.shape[0]

    minMAPEImprovmentForOutlier = options['minMAPEImprovmentForOutlier']
    outlierIndexes = []
    
    model = Forecast()
    fdict = model.forecast(frame, options.copy())
    baseMAPE = fdict['MAPE']
    
    for idx in range(rowCount):
        tmp = frame.copy()
        tmp['OUTLIER'] = 0
        tmp.at[idx, 'OUTLIER'] = 1
        tmpOptions = options.copy()
        tmpOptions['outlierColumn'] = 'OUTLIER'
        model = Forecast()
        fdict = model.forecast(tmp, tmpOptions)
        tmpMAPE = fdict['MAPE']
#         print('base MAPE:', baseMAPE, 'test MAPE:', tmpMAPE, 'minMAPEImprovmentForOutlier:', minMAPEImprovmentForOutlier)
        if baseMAPE >= (tmpMAPE + minMAPEImprovmentForOutlier):
            outlierIndexes.append(idx)
    
#     print(outlierIndexes)
    
    frame['OUTLIER'] = 0
    for idx in outlierIndexes:
        frame.at[idx, 'OUTLIER'] = 1

    return frame

def findMAPE(frame, options, seasonality):
    options = options.copy()
    options['seasonality'] = seasonality
    model = Forecast()
    return  model.forecast(frame.copy(), options)['MAPE']

def findOptimalSeasonality(frame, options):
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
  
#     fileName = 'c:/temp/retail_unit_demand_with_outliers.csv'
#     fileName = 'c:/temp/retail_unit_demand.csv'
#     jsonFileName = 'c:/temp/retail_unit_demand_options.json'
#     outputFileName = 'c:/temp/retail_unit_demand_forecast.csv'
#     buildSampleoptionsJSONFile(jsonFileName)
    
    if (len(sys.argv) < 3):
        print("Error: Insufficient arguments")
        sys.exit(-1)
        
    
    jsonFileName = sys.argv[1]
    fileName = sys.argv[2]
    outputFileName = sys.argv[3]
    
    with open(jsonFileName, 'r') as fp:
        options = json.load(fp)
    
#     print(options)
    
    frame = pd.read_csv(fileName)
        
    frame.sort_values(by=options['sortColumns'], ascending=True, inplace=True)
    
    frames = list(frame.groupby(by=options['splitColumns']))
    
    outputFrame = None

    for frame in frames:
        frame = frame[1]
        frame.reset_index(drop=True, inplace=True)
        
        if options['seasonality'] == 'Auto':
            options['seasonality'] = findOptimalSeasonality(frame.copy(), options.copy())
        
        if ('autoDetectOutliers' in options and options['autoDetectOutliers']):
            frame = detectOutliers(frame, options.copy())
            options['outlierColumn'] = 'OUTLIER'
        
        model = Forecast()
        fdict = model.forecast(frame, options.copy())
        frame = fdict['frame']
        
        outputFrame = frame if outputFrame is None else outputFrame.append(frame)
    
    outputFrame.to_csv(outputFileName, index=False)
    
    print("Forecast(s) complete...")
