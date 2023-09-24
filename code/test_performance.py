import numpy as np
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0],enable=True)
from tensorflow import keras
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Bidirectional
import matplotlib.pyplot as plt
import pickle

def test_perfomance():
    x_test=np.load("train_data/X_data_test.npy")
    y_test=np.load("train_data/Y_data_test.npy")
    scalerfile = 'model/scaler_y.pkl'
    scaler_y = pickle.load(open(scalerfile, 'rb'))
    model=tf.keras.models.load_model('model/model_nas_100.h5')
    predictions=model.predict(x_test)
    y_close=scaler_y.inverse_transform(y_test)
    y_pred = scaler_y.inverse_transform(predictions)
    cnt_right = 0
    cnt_wrong = 0 
    test_size=len(y_close)-1
    for i in range(test_size):
        if (y_close[i+1]-y_close[i])*(y_pred[i+1]-y_pred[i])>0:
            cnt_right+=1
        else:
            cnt_wrong+=1
    print("Vorzeichen in: ", 100*cnt_right/float(test_size), "% richtig")
    
    predicted_performances=(y_pred[:-1]-y_pred[1:])/y_pred[:-1]*100
    real_performances = (y_close[:-1]-y_close[1:])/y_close[:-1]*100
    true_cases=[]
    trade_cases=[]
    margins=np.linspace(0,20,80)

    for margin in margins:
        real_pos=set(np.where(real_performances>0)[0])
        pred_pos=set(np.where(predicted_performances>margin)[0])
        true_pos=len(pred_pos & real_pos)
        real_neg=set(np.where(real_performances<0)[0])
        pred_neg=set(np.where(predicted_performances<-margin)[0])
        true_neg=len(pred_neg & real_neg)
        true_cases.append(true_pos+true_neg)
        trade_cases.append(len(np.where(np.abs(predicted_performances) > margin)[0]))
    

    fig, ax = plt.subplots(1,2)
    np.seterr(divide='ignore', invalid='ignore')
    true_percentage = np.array(true_cases)/np.array(trade_cases)
    trade_percentage=np.array(trade_cases)/test_size
    ax[0].plot(margins, trade_percentage, label="Trade cases")
    ax[0].plot(margins, true_percentage, label="True percentage")
    mask = np.ma.masked_less(np.array(true_percentage, dtype=float), 0.5)
    ax[0].plot(margins, mask, "g")
    ax[1].plot(range(test_size+1), y_close, label="real close")
    ax[1].plot(range(test_size+1), y_pred , label="predict close")
    ax[0].axhline(y=0.5, color='r', linestyle='-', label="y=0.5")
    ax[0].set_xlabel("Safety margin")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Kurs")
    plt.legend()
    ax[0].grid()
    
    fig.savefig("figures/backtest_dax.png")
    return 0

if __name__=="__main__":
    test_perfomance()



