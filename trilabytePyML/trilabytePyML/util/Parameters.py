
def getParam(param, options):
    val = None if not(param in options) else options[param]
    
    if (val == None):
        defaults = dict()
        defaults['sortColumns'] = []
        defaults['splitColumns'] = []
        defaults['predictorColumns'] = []
        defaults['targetColumn'] = None 
        defaults['periodicity'] = 12
        defaults['seasonality'] = 'Multiplicative' #'Auto'  # 'Auto','None','Additive','Multiplicative' 
        defaults['method'] = 'MLR' #'ARIMA','MLR','Prophet'
        defaults['timestampColumn'] = "CAL_DATE" #only for FBProphet forecast
        defaults['autoDetectOutliers'] = False
        defaults['outlierStdevMultiplier'] = 3.0
        defaults['seasonalityBandwidth'] = 0.6
        defaults['ridgeAlpha'] = 1.0
        defaults['forceNonNegative'] = False
        
        val = None if not(param in defaults) else defaults[param]
        
        print("WARNING: ", param, " not found.  Assuming default: ", val)
    
    return val