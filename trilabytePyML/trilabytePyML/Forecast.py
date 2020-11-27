#
#    http://trilabyte.com
#    Trilabyte Python Machine Learning
#    Copyright (c) 2020 - Trilabyte
#    Author: Scott Mutchler
#    Contact: smutchler@trilabyte.com
#

import math
import operator
import random
from statistics import mean 
from statistics import median

from scipy import stats
from sklearn.linear_model import Ridge

import loess.loess_1d as lo
import numpy as np
import pmdarima as pm
from trilabytePyML.stats.Statistics import calcMAPE
from trilabytePyML.stats.Statistics import calcPredictionInterval


class Forecast:
    
    def preOutlierDetection(self, frame, options):
        targetColumn = options['targetColumn'] 
        
        frame['X_INDEX'] = frame.index.values
        frame['X_INTERPOLATED'] = frame[targetColumn]
        
        # split the data into past/future based on null in target column 
        nullIdx = frame[targetColumn].isnull()
        futureData = frame[nullIdx]
        historicalIdx = list(map(operator.not_, nullIdx))
        historicalData = frame[historicalIdx] 
         
        x = np.asarray(historicalData['X_INDEX'].tolist())    
        y = np.asarray(historicalData[options['targetColumn']].tolist())
        bandwidth = options['seasonalityBandwidth']
        xout, yout, weights = lo.loess_1d(x, y, frac=bandwidth, degree=2)
        
        frame['X_TREND'] = np.append(yout, np.asarray(futureData[targetColumn].tolist())) 
        frame['X_TREND_DIFF'] = frame[targetColumn] - frame['X_TREND']
        
        stdev = frame['X_TREND_DIFF'].std()
        avg = frame['X_TREND_DIFF'].mean()
        
#         print('stdev:',stdev, 'mean:',avg)
        
        mult = 3.0 
        if 'outlierStdevMultiplier' in options:
            mult = options['outlierStdevMultiplier'] 
        
        frame['X_OUTLIER'] = 0
        for index, row in frame.iterrows():
            diff = abs(frame['X_TREND_DIFF'][index])
            if diff > avg + mult * stdev:
                if index > 0 and index <= frame.shape[0] - 1:
                    frame['X_INTERPOLATED'][index] = mean([frame['X_INTERPOLATED'][index - 1], frame['X_INTERPOLATED'][index + 1]])
                    frame['X_OUTLIER'][index] = 1
                else:
                    frame['X_INTERPOLATED'][index] = frame['X_TREND'][index]
                    frame['X_OUTLIER'][index] = 1
        
        frame.drop(columns=['X_TREND', 'X_TREND_DIFF', 'X_INDEX'])
        
        fdict = dict()
        fdict['frame'] = frame
        fdict['options'] = options
        
        return fdict

    def prepare(self, frame, options):
        # create copy of target for modification (fill zeros with very small number)
        random.seed(158923)
        targetColumn = options['targetColumn']
        
        frame['X_INDEX'] = frame.index.values
        
        # ensure predictors and target are float
        frame[targetColumn] = frame[targetColumn].astype(float)
        frame[options['predictorColumns']] = frame[options['predictorColumns']].astype(float) 

        newTargetColumn = 'X_' + targetColumn
        
        # if we have done outlier detection there will be an interpolated column that has the interpolated actuals
        if 'X_INTERPOLATED' in frame:
            frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame['X_INTERPOLATED']))
        else:
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
        
        # short-circuit if none is chosen
        if fdict['options']['seasonality'] == 'None':
            frame['X_SEASONALITY'] = None
            fdict['frame'] = frame
            return fdict;
        
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
        
        # print("Using raw mean/median diff values for seasonality")
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
        if options['autoDetectOutliers']:
            fdict = self.preOutlierDetection(frame, options)
            frame = fdict['frame']
        
        fdict = self.prepare(frame, options)
        
        fdict = self.predictTrend(fdict)
                
        fdict = self.calcSeasonality(fdict)
        
        frame['X_SEASONALITY_TYPE'] = fdict['options']['seasonality']
        
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
        
        targetColumn = options['targetColumn']
        frame['X_APE'] = None 
        for index, row in frame.iterrows():
            frame['X_APE'][index] = (abs(row['X_FORECAST'] - row[targetColumn]) / row[targetColumn] * 100.0) if row[targetColumn] != 0 else None
        
        if 'forceNonNegative' in fdict['options'] and fdict['options']['forceNonNegative']:
            frame.loc[frame['X_FORECAST'] < 0, 'X_FORECAST'] = 0
            frame.loc[frame['X_UPI'] < 0, 'X_UPI'] = 0
            frame.loc[frame['X_LPI'] < 0, 'X_LPI'] = 0
        
        return fdict

    def forecastARIMA(self, frame, options):
        if options['autoDetectOutliers']:
            fdict = self.preOutlierDetection(frame, options)
            frame = fdict['frame']
        
        targetColumn = options['targetColumn']
        newTargetColumn = 'X_' + targetColumn
        options['targetColumn'] = newTargetColumn
        # if we have done outlier detection there will be an interpolated column that has the interpolated actuals
        if 'X_INTERPOLATED' in frame:
            frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame['X_INTERPOLATED']))
        else:
            frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame[options['targetColumn']]))    
        
        # split the data into past/future based on null in target column 
        nullIdx = frame[targetColumn].isnull()
        futureData = frame[nullIdx]
        historicalIdx = list(map(operator.not_, nullIdx))
        historicalData = frame[historicalIdx] 
         
        y = np.asarray(historicalData[newTargetColumn].tolist())
        
        model = pm.auto_arima(y, seasonal=True,
                     stepwise=True, suppress_warnings=True,
                     error_action='ignore')

        histPreds, histConf_int = model.predict_in_sample(return_conf_int=True)
        preds, conf_int = model.predict(n_periods=len(futureData), return_conf_int=True)

        forecast = np.concatenate((histPreds, preds))
        lpi = np.concatenate((list(map(lambda x: x[0], histConf_int)), list(map(lambda x: x[0], conf_int))))
        upi = np.concatenate((list(map(lambda x: x[1], histConf_int)), list(map(lambda x: x[1], conf_int))))
        
        frame['X_LPI'] = lpi
        frame['X_FORECAST'] = forecast 
        frame['X_UPI'] = upi 
        
        mape = calcMAPE(frame['X_FORECAST'], frame[targetColumn])
        frame['X_MAPE'] = mape
        fdict['MAPE'] = mape
        
        frame['X_RESIDUAL'] = frame['X_FORECAST'] - frame[targetColumn] 

        frame['X_APE'] = None 
        for index, row in frame.iterrows():
            frame['X_APE'][index] = (abs(row['X_FORECAST'] - row[targetColumn]) / row[targetColumn] * 100.0) if row[targetColumn] != 0 else None
        
        if 'forceNonNegative' in fdict['options'] and options['forceNonNegative']:
            frame.loc[frame['X_FORECAST'] < 0, 'X_FORECAST'] = 0
            frame.loc[frame['X_UPI'] < 0, 'X_UPI'] = 0
            frame.loc[frame['X_LPI'] < 0, 'X_LPI'] = 0
        
        fdict['frame'] = frame
        return fdict 

