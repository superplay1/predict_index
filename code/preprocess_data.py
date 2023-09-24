import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

def moving_std(data, window_size):
    ma = moving_average(data, window_size)
    squared_diff = (data[window_size - 1:] - ma) ** 2
    cum_squared_diff = np.cumsum(squared_diff, dtype=float)
    cum_squared_diff[window_size:] = cum_squared_diff[window_size:] - cum_squared_diff[:-window_size]
    return np.sqrt(cum_squared_diff / window_size)

def preprocess_train_data(df, lb, scalerx=None, scalery=None):
    ###scaling
    symbols = df["symbol"].drop_duplicates().to_list()
    df_new= pd.DataFrame()
    df_new["date"]=df.loc[df['symbol'] == "^GDAXI", 'date']
    df_new=df_new.set_index(df_new["date"]).drop("date", axis=1)
    for symbol in symbols:
        values=df.loc[df['symbol'] == symbol, 'close'].values
        df_new[symbol]=values
        # aver=moving_average(values, window)
        # std=moving_std(values, window)
        # y.append((values-aver)/std)
    if scalerx:
        scaled_features=scalerx.transform(df_new)
    else:
        scalerx=StandardScaler()
        scaled_features=scalerx.fit_transform(df_new)
    if scalery:
        scaled_features_y=scalery.transform(df_new["^GDAXI"].values.reshape(-1,1))
    else:
        scalery=StandardScaler()
        scaled_features_y=scalery.fit_transform(df_new["^GDAXI"].values.reshape(-1,1))

    df_scaled = pd.DataFrame(scaled_features, index=df_new.index, columns=df_new.columns)

    ###setting time series format
    X=[]
    Y=pd.DataFrame()
    for i in range(0, df_scaled.shape[0] - lb):
        X.append([])
        train = df_scaled.iloc[i : lb + i]
        goal = df_scaled.iloc[lb + i]
        X[i] = train.values
        Y = pd.concat([Y, pd.Series(goal.values[0])])
    X=np.array(X)
    Y=Y.values
    return scalerx, scalery, X, Y


if __name__== "__main__":
    df = pd.read_csv("train_data/test_dax.csv")
    scalerfile_x = 'model/scaler_x.pkl'
    scalerfile_y = 'model/scaler_y.pkl'
    scalerx=None
    scalery=None
    scalerx = pickle.load(open(scalerfile_x, 'rb'))
    scalery = pickle.load(open(scalerfile_y, 'rb'))
    scalerx, scalery, X, Y = preprocess_train_data(df, lb=8, scalerx=scalerx, scalery=scalery)
    pickle.dump(scalerx, open("model/scaler_x.pkl", 'wb'))
    pickle.dump(scalery, open("model/scaler_y.pkl", 'wb'))
    np.save("train_data/X_data_test", X)
    np.save("train_data/Y_data_test", Y)







