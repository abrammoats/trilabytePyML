1. Install Anaconda 64 bit
2. Open Anaconda prompt

pip install pandas loess scipy numpy scikit-learn pmdarima
conda install -c conda-forge ephem pystan fbprophet


# due to incompatability 12/18/2020
conda install pystan=2.19.0.0
conda install -c conda-forge fbprophet=0.6.0

# or try this
conda remove pystan fbprophet
conda install --channel conda-forge pystan fbprophet