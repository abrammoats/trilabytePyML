#
#    http://trilabyte.com
#    Trilabyte Python Machine Learning
#    Copyright (c) 2020 - Trilabyte
#    Author: Scott Mutchler
#    Contact: smutchler@trilabyte.com
#

import operator
import random
import math
import numpy as np
from statistics import median
import loess.loess_1d as lo
from sklearn.linear_model import Ridge
from trilabytePyML.stats.Statistics import calcMAPE
from trilabytePyML.stats.Statistics import calcPredictionInterval
from scipy import stats


class Forecast:

    def prepare(self, frame, options):
        # create copy of target for modification (fill zeros with very small number)
        random.seed(158923)
        targetColumn = options['targetColumn']
        
        frame['X_INDEX'] = frame.index.values
        
        # ensure predictors and target are float
        frame[targetColumn] = frame[targetColumn].astype(float)
        frame[options['predictorColumns']] = frame[options['predictorColumns']].astype(float) 

        newTargetColumn = 'X_' + targetColumn
        frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame[options['targetColumn']]))
        options['targetColumn'] = newTargetColumn
               
        # split the data into past/future based on null in target column 
        nullIdx = frame[targetColumn].isnull()
        futureData = frame[nullIdx]
        historicalIdx = list(map(operator.not_, nullIdx))
        historicalData = frame[historicalIdx] 
         
        x = np.asarray(historicalData['X_INDEX'].tolist())    
        y = np.asarray(historicalData[newTargetColumn].tolist())
        bandwidth = options['seasonalityBandwidth']
        xout, yout, weights = lo.loess_1d(x, y, frac=bandwidth, degree=2)
        
        frame['X_TREND'] = np.append(yout, np.asarray(futureData[targetColumn].tolist())) 
        frame['X_TREND_DIFF'] = frame[targetColumn] - frame['X_TREND']
        frame['X_TREND_RATIO'] = frame[targetColumn] / frame['X_TREND']
          
        fdict = dict()
        fdict['historicalIdx'] = historicalIdx
        fdict['futureIdx'] = nullIdx
        fdict['frame'] = frame
        fdict['options'] = options

        return fdict

    def predictTrend(self, fdict):
        frame = fdict['frame']
        historicalData = frame[fdict['historicalIdx']]
          
        if ('outlierColumn' in fdict['options'] and fdict['options']['outlierColumn'] in historicalData):
            historicalData = historicalData[historicalData[fdict['options']['outlierColumn']] == 0]
            
        x = historicalData[fdict['options']['predictorColumns']]
        y = historicalData['X_TREND']
                
        model = Ridge(alpha=fdict['options']['ridgeAlpha'])
        model.fit(x, y)
        
        xscore = frame[fdict['options']['predictorColumns']]
        yhat = model.predict(xscore)
        
        frame['X_TREND_PREDICTED'] = yhat
        
        return fdict

    def calcSeasonality(self, fdict):
        frame = fdict['frame']
        historicalData = frame[fdict['historicalIdx']] 
        periodicity = fdict['options']['periodicity']
        
        diffData = historicalData['X_TREND_DIFF' if fdict['options']['seasonality'] == 'Additive' else 'X_TREND_RATIO']
        
        trendCol = 'X_TREND' if not('adjustSeasonalityTrendColumn' in fdict['options']) else fdict['options']['adjustSeasonalityTrendColumn']
        trendData = historicalData[trendCol]
                        
        buckets = []  # holds trend (diff/ratio) data indexed by periodicity
        tbuckets = []  # holds corresponding trend data
        for idx in range(len(diffData)):
            diffVal = diffData.iloc[idx]
            trendVal = trendData.iloc[idx]

            outlier = ('outlierColumn' in fdict['options'] and fdict['options']['outlierColumn'] in historicalData and historicalData[fdict['options']['outlierColumn']].iloc[idx] == 1)
                
            bucketIdx = math.floor(idx % periodicity)
            if (not(outlier)):
                if (len(buckets) < (bucketIdx + 1)):
                    tbuckets.append([trendVal])
                    buckets.append([diffVal])
                else:
                    tbuckets[bucketIdx].append(trendVal)
                    buckets[bucketIdx].append(diffVal)
        
        seasonality = []
        rowCount = frame.shape[0]
        
        if ('adjustSeasonalityBasedOnTrend' in fdict['options'] and fdict['options']['adjustSeasonalityBasedOnTrend']):
            #print("Adjusting seasonality based on: ", trendCol)
            models = []
            # regress each bucket against the corresponding 
            for idx in range(0, len(buckets)):
                y = buckets[idx]
                x = tbuckets[idx]
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                models.append([intercept, slope, r_value, idx])  
           
            for idx in range(rowCount):
                diffIdx = math.floor(idx % periodicity)
                model = models[diffIdx]

                if trendCol == 'X_TREND' and math.isnan(frame['X_TREND'][idx]):
                    x = frame['X_TREND_PREDICTED'][idx]
                else:
                    x = frame[trendCol][idx]
                
                #print('idx:',idx,'diffIdx:',diffIdx,'intercept:',model[0],'slope:',model[1],'x:',x,'yhat:',model[0] + model[1] * x)
                seasonality.append(model[0] + model[1] * x)
        else:
            #print("Using raw median diff values for seasonality")
            medianVals = []
            for diffs in buckets:
                medianVals.append(median(diffs))
            
            for idx in range(rowCount):
                diffIdx = math.floor(idx % periodicity)
                seasonality.append(medianVals[diffIdx])

        frame['X_SEASONALITY'] = seasonality
        fdict['frame'] = frame
        
        return fdict
       
    def forecast(self, frame, options):
        fdict = self.prepare(frame, options)
        
        fdict = self.predictTrend(fdict)
                
        fdict = self.calcSeasonality(fdict)
        
        if (fdict['options']['seasonality'] == 'Additive'):
            frame['X_FORECAST'] = frame['X_SEASONALITY'] + frame['X_TREND_PREDICTED']
        elif (fdict['options']['seasonality'] == 'Multiplicative'):
            frame['X_FORECAST'] = frame['X_SEASONALITY'] * frame['X_TREND_PREDICTED']
        else:
            frame['X_FORECAST'] = frame['X_TREND_PREDICTED']
        
        mape = calcMAPE(frame['X_FORECAST'], frame[fdict['options']['targetColumn']])
        frame['X_MAPE'] = mape
        fdict['MAPE'] = mape
        
        frame['X_RESIDUAL'] = frame['X_FORECAST'] - frame[options['targetColumn']] 
        
        predictionInterval = calcPredictionInterval(frame['X_RESIDUAL'])

        frame['X_LPI'] = frame['X_FORECAST'] - predictionInterval 
        frame['X_UPI'] = frame['X_FORECAST'] + predictionInterval 
        
        return fdict
        
#         
#       
        
