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
import trilabytePyML.util.Parameters as params 
from fbprophet import Prophet
import pandas as pd
from sklearn.model_selection import ParameterGrid
import json


class Forecast:
    
    def lastNonNullIndex(self, x):
        idx = 0
        for i in range(len(x)):
            if not(np.isnan(x[i])):
                idx = i
        
        return idx
    
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
        y = np.asarray(historicalData[params.getParam('targetColumn', options)].tolist())
        bandwidth = params.getParam('seasonalityBandwidth', options)
        xout, yout, weights = lo.loess_1d(x, y, frac=bandwidth, degree=2)
        
        frame['X_TREND'] = np.append(yout, np.asarray(futureData[targetColumn].tolist())) 
        frame['X_TREND_DIFF'] = frame[targetColumn] - frame['X_TREND']
        
        stdev = frame['X_TREND_DIFF'].std()
        avg = frame['X_TREND_DIFF'].mean()
        
        mult = params.getParam('outlierStdevMultiplier', options)
        
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
        targetColumn = params.getParam('targetColumn', options)
        
        frame['X_INDEX'] = frame.index.values
        
        # scale predictors between 0 and 1 
        try:
            newPredCols = []
            if params.getParam('scalePredictors', options):
                for predCol in params.getParam('predictorColumns', options):
                    newCol = 'X_' + predCol 
                    frame[newCol] = (frame[predCol] - frame[predCol].min()) / (frame[predCol].max() - frame[predCol].min())
                    newPredCols.append(newCol)
                options['predictorColumns'] = newPredCols
        except Exception as e:
            print("Unable to scale predictors: ", e)
        
        # ensure predictors and target are float
        frame[targetColumn] = frame[targetColumn].astype(float)
        frame[params.getParam('predictorColumns', options)] = frame[params.getParam('predictorColumns', options)].astype(float) 

        newTargetColumn = 'X_' + targetColumn
        
        # if we have done outlier detection there will be an interpolated column that has the interpolated actuals
        if 'X_INTERPOLATED' in frame:
            frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame['X_INTERPOLATED']))
        else:
            frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame[params.getParam('targetColumn', options)]))
        
        options['targetColumn'] = newTargetColumn
        
        # split the data into past/future based on null in target column 
        lastNonNullIdx = self.lastNonNullIndex(frame[targetColumn])
        fullHistoricalIdx = frame['X_INDEX'] <= lastNonNullIdx
        
        # we use full history for trending/smoothing (but NOT modeling in the future)
        fullHistoricalData = frame[fullHistoricalIdx]
        numHoldoutRows = params.getParam('numHoldoutRows', options)
        fullFutureIdx = frame['X_INDEX'] > lastNonNullIdx
        fullFutureData = frame[fullFutureIdx]
        
        # we store history minus hold-out for future modeling
        lastNonNullIdx = lastNonNullIdx - numHoldoutRows
        historicalIdx = frame['X_INDEX'] <= lastNonNullIdx
        #historicalData = frame[historicalIdx]
        futureIdx = frame['X_INDEX'] > lastNonNullIdx
        #futureData = frame[futureIdx]
        
        if (numHoldoutRows > 0):
            evalIdx = list(map(lambda x: x > lastNonNullIdx and x <= (lastNonNullIdx + numHoldoutRows), frame['X_INDEX']))
        else:
            evalIdx = historicalIdx
         
        x = np.asarray(fullHistoricalData['X_INDEX'].tolist())    
        y = np.asarray(fullHistoricalData[newTargetColumn].tolist())
        bandwidth = params.getParam('seasonalityBandwidth', options)
        xout, yout, weights = lo.loess_1d(x, y, frac=bandwidth, degree=2)
        
        frame['X_TREND'] = np.append(yout, np.asarray(fullFutureData[targetColumn].tolist())) 
        frame['X_TREND_DIFF'] = frame[targetColumn] - frame['X_TREND']
        frame['X_TREND_RATIO'] = frame[targetColumn] / frame['X_TREND']
          
        fdict = dict()
        fdict['historicalIdx'] = historicalIdx
        fdict['futureIdx'] = futureIdx
        fdict['evalIdx'] = evalIdx
        fdict['frame'] = frame
        fdict['options'] = options

        return fdict

    def predictTrend(self, fdict):
        frame = fdict['frame']
        historicalData = frame[fdict['historicalIdx']]
        options = fdict['options']
                      
        x = historicalData[params.getParam('predictorColumns', options)]
        y = historicalData['X_TREND']
                
        model = Ridge(alpha=params.getParam('ridgeAlpha', options))
        model.fit(x, y)
        
        xscore = frame[params.getParam('predictorColumns', options)]
        yhat = model.predict(xscore)
        
        frame['X_PREDICTORS'] = ','.join(params.getParam('predictorColumns', options))
        frame['X_COEFFICIENTS'] = ','.join(map(str, model.coef_))
        frame['X_INTERCEPT'] = str(model.intercept_)
        
        frame['X_TREND_PREDICTED'] = yhat
        
        return fdict

    def calcSeasonality(self, fdict):
        frame = fdict['frame']
        historicalData = frame[fdict['historicalIdx']] 
        options = fdict['options']
        periodicity = params.getParam('periodicity', options)
        
        # short-circuit if none is chosen
        if params.getParam('seasonality', options) == 'None':
            frame['X_SEASONALITY'] = None
            fdict['frame'] = frame
            return fdict;
        
        diffData = historicalData['X_TREND_DIFF' if params.getParam('seasonality', options) == 'Additive' else 'X_TREND_RATIO']
        
        trendData = historicalData['X_TREND']
                        
        buckets = []  # holds trend (diff/ratio) data indexed by periodicity
        tbuckets = []  # holds corresponding trend data
        for idx in range(len(diffData)):
            diffVal = diffData.iloc[idx]
            trendVal = trendData.iloc[idx]

            bucketIdx = math.floor(idx % periodicity)
            if (len(buckets) < (bucketIdx + 1)):
                tbuckets.append([trendVal])
                buckets.append([diffVal])
            else:
                tbuckets[bucketIdx].append(trendVal)
                buckets[bucketIdx].append(diffVal)
        
        seasonality = []
        rowCount = frame.shape[0]
        
        medianVals = []
        for diffs in buckets:
            medianVals.append(median(diffs))
        
        for idx in range(rowCount):
            diffIdx = math.floor(idx % periodicity)
            seasonality.append(medianVals[diffIdx])

        frame['X_SEASONALITY'] = seasonality
        fdict['frame'] = frame
        
        return fdict
       
    def forecastMLR(self, frame, options):
        predCols = params.getParam('predictorColumns', options)
        if len(predCols) == 0:
            predCols.append('X_INDEX')
            options['predictorColumns'] = predCols 
    
        if params.getParam('autoDetectOutliers', options):
            fdict = self.preOutlierDetection(frame, options)
            frame = fdict['frame']
        
        fdict = self.prepare(frame, options)
        
        fdict = self.predictTrend(fdict)
                
        fdict = self.calcSeasonality(fdict)
        
        options = fdict['options']
        frame['X_SEASONALITY_TYPE'] = params.getParam('seasonality', options)
                
        if (params.getParam('seasonality', options) == 'Additive'):
            frame['X_FORECAST'] = frame['X_SEASONALITY'] + frame['X_TREND_PREDICTED']
        elif (params.getParam('seasonality', options) == 'Multiplicative'):
            frame['X_FORECAST'] = frame['X_SEASONALITY'] * frame['X_TREND_PREDICTED']
        else:
            frame['X_FORECAST'] = frame['X_TREND_PREDICTED']
            
        evalIdx = fdict['evalIdx']
        targetColumn = params.getParam('targetColumn', options)
        evalFrame = frame[evalIdx]
        mape = calcMAPE(evalFrame['X_FORECAST'], evalFrame[targetColumn])
        frame['X_MAPE'] = mape
        fdict['MAPE'] = mape
        
        frame['X_RESIDUAL'] = frame['X_FORECAST'] - frame[params.getParam('targetColumn', options)] 
        
        predictionInterval = calcPredictionInterval(frame['X_RESIDUAL'])

        frame['X_LPI'] = frame['X_FORECAST'] - predictionInterval 
        frame['X_UPI'] = frame['X_FORECAST'] + predictionInterval 
        
        targetColumn = params.getParam('targetColumn', options)
        frame['X_APE'] = None 
        for index, row in frame.iterrows():
            frame['X_APE'][index] = (abs(row['X_FORECAST'] - row[targetColumn]) / row[targetColumn] * 100.0) if row[targetColumn] != 0 else None
        
        if params.getParam('forceNonNegative', options):
            frame.loc[frame['X_FORECAST'] < 0, 'X_FORECAST'] = 0
            frame.loc[frame['X_UPI'] < 0, 'X_UPI'] = 0
            frame.loc[frame['X_LPI'] < 0, 'X_LPI'] = 0
        
        frame['X_HYPERTUNE'] = None
        
        return fdict

    def forecastProphetInternal(self, frame, options, pframe, historicalData, futureData, seasonalityMode, intervalWidth, changePointPriorScale, holidayPriorScale, changePointFraction):
        nChangePoints = math.ceil(len(historicalData) * changePointFraction)
        
        periodicity = params.getParam('periodicity', options) 
        
        weeklySeasonality = (periodicity == 52 or periodicity == 53)
        dailySeasonality = (periodicity == 356)
        
        model = Prophet(yearly_seasonality=True, daily_seasonality=dailySeasonality, weekly_seasonality=weeklySeasonality, interval_width=intervalWidth, seasonality_mode=seasonalityMode, changepoint_prior_scale=changePointPriorScale, holidays_prior_scale=holidayPriorScale, n_changepoints=nChangePoints)
        
        for pred in params.getParam('predictorColumns', options):
            model.add_regressor(pred)
            pframe[pred] = historicalData[pred]
        
        model.fit(pframe)

        #
        # Frequencies are defined here:
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        #
        freq = None
        if (periodicity == 12):
            freq = 'MS'  # monthly start
        elif (periodicity == 365):
            freq = 'D'  # daily
        elif (periodicity == 4):
            freq = 'QS'  # quarterly start
        elif (periodicity == 52 or periodicity == 53):
            freq = 'W'  # weekly

        future = model.make_future_dataframe(periods=len(futureData), freq=freq)
        
        for pred in params.getParam('predictorColumns', options):
            future[pred] = frame[pred]
        
        forecast = model.predict(future)
        
        return forecast 

    def forecastProphet(self, frame, options):
        targetColumn = params.getParam('targetColumn', options)
        newTargetColumn = 'X_' + targetColumn
        
        if params.getParam('autoDetectOutliers', options):
            fdict = self.preOutlierDetection(frame, options)
            frame = fdict['frame']
        
        # if we have done outlier detection there will be an interpolated column that has the interpolated actuals
        if 'X_INTERPOLATED' in frame:
            frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame['X_INTERPOLATED']))
        else:
            frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame[params.getParam('targetColumn', options)]))    
        
        options['targetColumn'] = newTargetColumn
        
        frame['X_INDEX'] = frame.index.values
        
        # split the data into past/future based on null in target column 
        lastNonNullIdx = self.lastNonNullIndex(frame[targetColumn])
        numHoldoutRows = params.getParam('numHoldoutRows', options)
        lastNonNullIdx = lastNonNullIdx - numHoldoutRows
        historicalIdx = frame['X_INDEX'] <= lastNonNullIdx
        historicalData = frame[historicalIdx]
        futureIdx = frame['X_INDEX'] > lastNonNullIdx
        futureData = frame[futureIdx]
        if (numHoldoutRows > 0):
            evalIdx = list(map(lambda x: x > lastNonNullIdx and x <= (lastNonNullIdx + numHoldoutRows), frame['X_INDEX']))
        else:
            evalIdx = historicalIdx
                
        pframe = pd.DataFrame()
        pframe['ds'] = historicalData[params.getParam('timestampColumn', options)]
        pframe['y'] = historicalData[params.getParam('targetColumn', options)]
        
        championPJSON = None
        if not(params.getParam('hypertune', options)):
            interval_width = params.getParam('interval_width', options)
            changepoint_prior_scale = params.getParam('changepoint_prior_scale', options)
            holidays_prior_scale = params.getParam('holidays_prior_scale', options)
            changepoints_fraction = params.getParam('changepoints_fraction', options)
            
            forecast = self.forecastProphetInternal(frame, options, pframe, historicalData, futureData, 'multiplicative', interval_width, changepoint_prior_scale, holidays_prior_scale, changepoints_fraction)
        else:
            forecast = None 
            championP = None
            championMAPE = 1E20
            
            # find best model through hyper-tuning
            pvals = {'seasonality_mode':('multiplicative', 'additive'),
               'changepoint_prior_scale':[0.05, 0.15, 0.25],
              'holidays_prior_scale':[0.1, 1.0, 10.0],
              'changepoint_fraction': [0.05, 0.1, 0.2]}
            
            pgrid = ParameterGrid(pvals)
            
            interval_width = params.getParam('interval_width', options)            
            
            for p in pgrid:
                challengerForecast = self.forecastProphetInternal(frame, options, pframe, historicalData, futureData, p['seasonality_mode'], interval_width, p['changepoint_prior_scale'], p['holidays_prior_scale'], p['changepoint_fraction'])
                evalFrame = challengerForecast[evalIdx]
                actualsFrame = frame[evalIdx]
                mape = calcMAPE(evalFrame['yhat'], actualsFrame[params.getParam('targetColumn', options)])
            
                if mape < championMAPE:
                    championP = p
                    championMAPE = mape 
                    forecast = challengerForecast
            
            championPJSON = json.dumps(championP)
            
            print('Champion parameters: ', championPJSON, 'Champion MAPE: ', championMAPE)
    
        frame['X_LPI'] = forecast['yhat_lower']
        frame['X_FORECAST'] = forecast['yhat']
        frame['X_UPI'] = forecast['yhat_upper']
        
        evalFrame = frame[evalIdx]
        mape = calcMAPE(evalFrame['X_FORECAST'], evalFrame[params.getParam('targetColumn', options)])
        frame['X_MAPE'] = mape
        
        frame['X_RESIDUAL'] = frame['X_FORECAST'] - frame[params.getParam('targetColumn', options)] 
        
        targetColumn = params.getParam('targetColumn', options)
        frame['X_APE'] = None 
        for index, row in frame.iterrows():
            frame['X_APE'][index] = (abs(row['X_FORECAST'] - row[targetColumn]) / row[targetColumn] * 100.0) if row[targetColumn] != 0 else None
        
        if 'forceNonNegative' in options and params.getParam('forceNonNegative', options):
            frame.loc[frame['X_FORECAST'] < 0, 'X_FORECAST'] = 0
            frame.loc[frame['X_UPI'] < 0, 'X_UPI'] = 0
            frame.loc[frame['X_LPI'] < 0, 'X_LPI'] = 0

        # add columns for consistency with other methods        
        frame['X_SEASONALITY'] = None 
        frame['X_SEASONALITY_TYPE'] = None 
        frame['X_TREND_PREDICTED'] = None 
        frame['X_TREND_RATIO'] = None  
        frame['X_PREDICTORS'] = None
        frame['X_COEFFICIENTS'] = None
        frame['X_INTERCEPT'] = None
        frame['X_HYPERTUNE'] = championPJSON
                
        fdict = dict()
        fdict['MAPE'] = mape
        fdict['frame'] = frame
            
        return fdict

    def forecastARIMA(self, frame, options):
        targetColumn = params.getParam('targetColumn', options)
        newTargetColumn = 'X_' + targetColumn
        
        if params.getParam('autoDetectOutliers', options):
            fdict = self.preOutlierDetection(frame, options)
            frame = fdict['frame']
        
        # if we have done outlier detection there will be an interpolated column that has the interpolated actuals
        if 'X_INTERPOLATED' in frame:
            frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame['X_INTERPOLATED']))
        else:
            frame[newTargetColumn] = list(map(lambda x: (x if x != 0.0 else random.random() / 1E5), frame[params.getParam('targetColumn', options)]))    
        
        options['targetColumn'] = newTargetColumn
        
        frame['X_INDEX'] = frame.index.values
        
        # split the data into past/future based on null in target column 
        lastNonNullIdx = self.lastNonNullIndex(frame[targetColumn])
        numHoldoutRows = params.getParam('numHoldoutRows', options)
        lastNonNullIdx = lastNonNullIdx - numHoldoutRows
        historicalIdx = frame['X_INDEX'] <= lastNonNullIdx
        historicalData = frame[historicalIdx]
        futureIdx = frame['X_INDEX'] > lastNonNullIdx
        futureData = frame[futureIdx]
        if (numHoldoutRows > 0):
            evalIdx = list(map(lambda x: x > lastNonNullIdx and x <= (lastNonNullIdx + numHoldoutRows), frame['X_INDEX']))
        else:
            evalIdx = historicalIdx
        
        y = np.asarray(historicalData[newTargetColumn].tolist())
        
        if len(params.getParam('predictorColumns', options)) > 0:
            x = historicalData[params.getParam('predictorColumns', options)]
            model = pm.auto_arima(y, exogenous=x, seasonal=True,
                     stepwise=not(params.getParam('hypertune', options)), suppress_warnings=True,
                     error_action='ignore')

            histPreds, histConf_int = model.predict_in_sample(exogenous=x, return_conf_int=True)
        
            x = futureData[params.getParam('predictorColumns', options)]
            preds, conf_int = model.predict(exogenous=x, n_periods=len(futureData), return_conf_int=True)
        else: 
            model = pm.auto_arima(y, seasonal=True,
                     stepwise=False, suppress_warnings=True,
                     error_action='ignore')

            histPreds, histConf_int = model.predict_in_sample(return_conf_int=True)

            preds, conf_int = model.predict(n_periods=len(futureData), return_conf_int=True)

        forecast = np.concatenate((histPreds, preds))
        lpi = np.concatenate((list(map(lambda x: x[0], histConf_int)), list(map(lambda x: x[0], conf_int))))
        upi = np.concatenate((list(map(lambda x: x[1], histConf_int)), list(map(lambda x: x[1], conf_int))))
        
        frame['X_LPI'] = lpi
        frame['X_FORECAST'] = forecast 
        frame['X_UPI'] = upi 
        
        evalFrame = frame[evalIdx]
        mape = calcMAPE(evalFrame['X_FORECAST'], evalFrame[targetColumn])
        frame['X_MAPE'] = mape
        
        frame['X_RESIDUAL'] = frame['X_FORECAST'] - frame[targetColumn] 

        frame['X_APE'] = None 
        for index, row in frame.iterrows():
            frame['X_APE'][index] = (abs(row['X_FORECAST'] - row[targetColumn]) / row[targetColumn] * 100.0) if row[targetColumn] != 0 else None
        
        if params.getParam('forceNonNegative', options):
            frame.loc[frame['X_FORECAST'] < 0, 'X_FORECAST'] = 0
            frame.loc[frame['X_UPI'] < 0, 'X_UPI'] = 0
            frame.loc[frame['X_LPI'] < 0, 'X_LPI'] = 0
        
        # add columns for consistency with other methods        
        frame['X_SEASONALITY'] = None 
        frame['X_SEASONALITY_TYPE'] = None 
        frame['X_TREND_PREDICTED'] = None 
        frame['X_TREND_RATIO'] = None 
        frame['X_HYPERTUNE'] = None 
        frame['X_PREDICTORS'] = None
        frame['X_COEFFICIENTS'] = None
        frame['X_INTERCEPT'] = None
        
        fdict = dict()
        fdict['frame'] = frame
        return fdict 

