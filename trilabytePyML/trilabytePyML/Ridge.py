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
from sklearn.linear_model import Ridge

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
        
def predict(frame, options):
    fdict = dict()
    
    roleCol = options['roleColumn'][0]
    trainFrame = frame.loc[frame[roleCol] == 'TRAIN']
    
    x = trainFrame[options['predictorColumns']]
    y = trainFrame[options['targetColumn']]
                
    model = Ridge(alpha=options['ridgeAlpha'])
    model.fit(x, y)
    
    xscore = frame[options['predictorColumns']]
    yhat = model.predict(xscore)
    
    frame['X_PREDICTED'] = yhat

    fdict['frame'] = frame
    return(fdict)

##############################
# Main
##############################
if __name__ == '__main__':
    
    print("Ridge - Stacked Data with Role Definition (TRAIN,SCORE)")
    print("-------------------------------")
    print("Required Librarires:")
    print("pip install pandas loess scipy numpy scikit-learn")
    print("-------------------------------")
    print("Usage: python -m trilabytePyML.Ridge [json options] [csv source data] [output csv file]")
    print("-------------------------------")
  
<<<<<<< HEAD
#     fileName = 'c:/temp/iris_with_role_and_split.csv'
#     jsonFileName = 'c:/temp/iris_ridge.json'
#     buildSampleoptionsJSONFile(jsonFileName)
#     outputFileName = 'c:/temp/iris_ridge_output.csv'
=======
    fileName = 'c:/temp/iris_with_role_and_split.csv'
    outputFileName = 'c:/temp/iris_ridge.csv'
    jsonFileName = 'c:/temp/iris_ridge.json'
    buildSampleoptionsJSONFile(jsonFileName)
>>>>>>> branch 'master' of https://github.com/smutchler/trilabytePyML
    
    jsonFileName = sys.argv[1]
    fileName = sys.argv[2]
    outputFileName = sys.argv[3]
    
    with open(jsonFileName, 'r') as fp:
        options = json.load(fp)
    
    print('Options:') 
    print(options,'\n')

    frame = pd.read_csv(fileName)
    frames = list(frame.groupby(by=options['splitColumns']))

    outputFrame = None
 
    for frame in frames:
        frame = frame[1]
        frame.reset_index(drop=True, inplace=True)
        
        fdict = predict(frame, options)
        frame = fdict['frame']
         
        outputFrame = frame if outputFrame is None else outputFrame.append(frame)
     
<<<<<<< HEAD
    outputFrame.to_csv(outputFileName, index=False)
     
    print("Forecast(s) complete...")
=======
    print(options)

    frame = pd.read_csv(fileName)
    frame.sort_values(by=options['splitColumns'], ascending=True, inplace=True)
     
    frames = list(frame.groupby(by=options['splitColumns']))
     
    outputFrame = None
 
    for frame in frames:
        frame = frame[1]
        frame.reset_index(drop=True, inplace=True)

        print(frame.head(10));
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
>>>>>>> branch 'master' of https://github.com/smutchler/trilabytePyML
