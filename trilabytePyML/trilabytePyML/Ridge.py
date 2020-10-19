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

def buildSampleoptionsJSONFile(jsonFileName):
    options = dict()
    options['splitColumns'] = ['Split']
    options['predictorColumns'] = ['Sepal.Width','Sepal.Length','Petal.Width']
    options['roleColumn'] = ['Role']
    options['targetColumn'] = 'Petal.Length' 
    options['ridgeAlpha'] = 1.0
    
    # print(json.dumps(options))
    
    with open(jsonFileName, 'w') as fp:
        json.dump(options, fp)

##############################
# Main
##############################
if __name__ == '__main__':
    
    print("Ridge")
    print("-------------------------------")
    print("Required Librarires:")
    print("pip install pandas loess scipy numpy scikit-learn")
    print("-------------------------------")
    print("Usage: python -m trilabytePyML.Ridge [json options] [csv source data] [output csv file]")
    print("-------------------------------")
  
#     fileName = 'c:/temp/retail_unit_demand_with_outliers.csv'
#     fileName = 'c:/temp/retail_unit_demand.csv'
#     outputFileName = 'c:/temp/retail_unit_demand_forecast.csv'
    jsonFileName = 'c:/temp/iris_ridge.json'
    buildSampleoptionsJSONFile(jsonFileName)
    
#     jsonFileName = sys.argv[1]
#     fileName = sys.argv[2]
#     outputFileName = sys.argv[3]
#     
    with open(jsonFileName, 'r') as fp:
        options = json.load(fp)
     
    print(options)
#     
#     frame = pd.read_csv(fileName)
#         
#     frame.sort_values(by=options['sortColumns'], ascending=True, inplace=True)
#     
#     frames = list(frame.groupby(by=options['splitColumns']))
#     
#     outputFrame = None
# 
#     for frame in frames:
#         frame = frame[1]
#         frame.reset_index(drop=True, inplace=True)
#         
#         if options['seasonality'] == 'Auto':
#             options['seasonality'] = findOptimalSeasonality(frame.copy(), options.copy())
#         
#         if ('autoDetectOutliers' in options and options['autoDetectOutliers']):
#             frame = detectOutliers(frame, options.copy())
#             options['outlierColumn'] = 'OUTLIER'
#         
#         model = Forecast()
#         fdict = model.forecast(frame, options.copy())
#         frame = fdict['frame']
#         
#         outputFrame = frame if outputFrame is None else outputFrame.append(frame)
#     
#     outputFrame.to_csv(outputFileName, index=False)
#     
#     print("Forecast(s) complete...")
