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

def create_network(x,y, epochs, name, index):
    train_x, validate_x, train_y, validate_y = train_test_split(x, y,
                                                    test_size=0.2, shuffle=True)
    model = tf.keras.Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2]))))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(16)))
    model.add(Dense(1))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanSquaredError())
                #metrics=['accuracy'])
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='model/'+str(name),
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=False,
        )

    history=model.fit(train_x, train_y, validation_data = (validate_x, validate_y),batch_size=64 ,epochs=epochs, callbacks=model_checkpoint_callback)
    fig, ax = plt.subplots(1,1)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_yscale('log')
    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig("figures/loss_"+index + ".png")
    best_loss = min(history.history['val_loss'])
    print("Best Validation loss:", best_loss)
    return model

def create_net(epochs):
    name='model_nas_100.h5'
    x_train = np.load('train_data/X_data.npy')
    y_train = np.load('train_data/Y_data.npy')
    model=create_network(x_train, y_train, epochs, name, "dax")
    tf.keras.backend.clear_session()
    return 0

def mute():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return original_stdout

def unmute(out):
    sys.stdout.close()
    sys.stdout = out

if __name__=="__main__":
    #out=mute()
    create_net(epochs=100)
    #unmute(out)

