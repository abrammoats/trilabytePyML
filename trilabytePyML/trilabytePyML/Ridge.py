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
    options['predictorColumns'] = ['Sepal.Width', 'Sepal.Length', 'Petal.Width']
    options['roleColumn'] = 'Role'
    options['targetColumn'] = 'Petal.Length' 
    options['ridgeAlpha'] = 1.0
    
    # print(json.dumps(options))
    
    with open(jsonFileName, 'w') as fp:
        json.dump(options, fp)

        
def predict(frame, options):
    fdict = dict()
    
    trainFrame = frame.loc[frame[options['roleColumn']] == 'TRAINING']
    
    x = trainFrame[options['predictorColumns']]
    y = trainFrame[options['targetColumn']]
                
    model = Ridge(alpha=options['ridgeAlpha'])
    model.fit(x, y)
    
    xscore = frame[options['predictorColumns']]
    yhat = model.predict(xscore)
    
    frame['X_PREDICTED'] = yhat

    fdict['frame'] = frame
    return(fdict)

def splitIntoFramesAndPredict(frame, options):
    frames = list(frame.groupby(by=options['splitColumns']))
    
    outputFrame = None
 
    for frame in frames:
        frame = frame[1]
        frame.reset_index(drop=True, inplace=True)
        
        fdict = predict(frame, options)
        frame = fdict['frame']
         
        outputFrame = frame if outputFrame is None else outputFrame.append(frame)
    
    return outputFrame

##############################
# Main
##############################
if __name__ == '__main__':
    
    print("Ridge - Stacked Data with Role Definition (TRAINING,SCORING)")
    print("-------------------------------")
    print("Required Librarires:")
    print("pip install pandas loess scipy numpy scikit-learn")
    print("-------------------------------")
    print("Usage: python -m trilabytePyML.Ridge [json options] [csv source data] [output csv file]")
    print("-------------------------------")

#     fileName = 'c:/temp/iris_with_role_and_split.csv'
#     outputFileName = 'c:/temp/iris_ridge.csv'
#     jsonFileName = 'c:/temp/iris_ridge.json'
    
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
    
    outputFrame = splitIntoFramesAndPredict(frame, options)
     
    outputFrame.to_csv(outputFileName, index=False)
     
    print("Output file: ", outputFileName)
     
    print("Predictions complete...")

