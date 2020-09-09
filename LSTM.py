import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
from keras.callbacks import EarlyStopping
import time
from sklearn import preprocessing

EPOCHS = 50  # how many passes through our data
BATCH_SIZE = 5  # how many batches? Try smaller batch if you"re getting OOM (out of memory) errors.
NAME = f"LSTM_TFT_PRED_{int(time.time())}"
FILE_NAME = "tft_matches.csv"
NUMBER_PLAYERS = 7

def preprocess_df(df):
    # this is a list of list that will CONTAIN the sequences and their target value
    all_data = []
    sequential_data = []
    target = []

    time_index = df.iloc[0].name
    for i, row in df.iterrows():
        if time_index == i:
            sequential_data.append([n for n in row[:NUMBER_PLAYERS]])
            target = [n for n in row[NUMBER_PLAYERS:]]
        else:
            time_index = i
            # everything but the last element since it will be the attack of the next game
            sequential_data = sequential_data[:-1]
            all_data.append([sequential_data, target])

            # resets the list
            sequential_data = []
            sequential_data.append([n for n in row[:NUMBER_PLAYERS]])
            target = [n for n in row[NUMBER_PLAYERS:]]

    # need to update the last entry since the time won't Change
    sequential_data = sequential_data[:-1]
    all_data.append([sequential_data, target])

    # shuffle for good measure
    random.shuffle(all_data)

    X = []
    y = []

    # splutting it back up to be passed into our model for training
    for seq, target in all_data:
        X.append(np.array(seq))
        y.append(np.array(target))

    return np.array(X), np.array(y)

# Since we are short data i will truncate the data by a few and make the shorter
# one for train/value
# i.e. 60/20/20
# 1, 2, 3, 4, 5, 6, 7, 8
# train = [1, 2, 3, 4] /val = [1, 2, 3, 4, 5, 6] / test = [1, 2, 3, 4, 5, 6, 7, 8]
# this way we wont hit the arrow of time where we are predicting on data we have already
# seen and we can increase out data size at the same time
def split_data(data):
    all_split_data = []
    new_y = []
    for i in data:
        # Truncates the data by 80% and adds it to the overall list
        index = int(len(i)*0.8)
        all_split_data.append(i[:index])
        new_y.append(i[index])

    return np.asarray(all_split_data), np.asarray(new_y)

def main():
    main_df = pd.DataFrame() # begin empty

    player_name = []
    target_name = []
    for i in range(NUMBER_PLAYERS):
        player_name.append(f"player_{i+1}")
        target_name.append(f"target_{i+1}")
    player_name.insert(0,"time")

    main_df = pd.read_csv(FILE_NAME, names=player_name)
    main_df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time

    # removes time from the list
    player_name.pop(0)

    # if there are gaps in data, use previously known values
    main_df.fillna(method="ffill", inplace=True)
    main_df.dropna(inplace=True)

    # shifts the data down so it can get predicted
    main_df[target_name] = main_df[player_name].shift(-1)

    main_df.dropna(inplace=True)

    X, y = preprocess_df(main_df)

    X = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post")

    validation_x, validation_y = split_data(X)
    train_x, train_y = split_data(validation_x)

    print(f"train data: {len(X)}")
    print(f"X.shape = {X.shape}")
    print(f"y.shape = {y.shape}")
    print(f"validation_x.shape = {validation_x.shape}")
    print(f"validation_y.shape = {validation_y.shape}")
    print(f"train_x.shape = {train_x.shape}")
    print(f"train_y.shape = {train_y.shape}")

    model = Sequential()
    model.add(LSTM(128, input_shape=(None,NUMBER_PLAYERS), activation="tanh", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, activation="tanh", return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(NUMBER_PLAYERS, activation="softmax"))

    model.summary()

    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    # checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")) # saves only the best ones

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    # Train model
    history = model.fit(
        np.asarray(train_x), np.asarray(train_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose = 1,
        validation_data=(np.asarray(validation_x), np.asarray(validation_y)),
        callbacks = [es]
    )

    # Score model
    score = model.evaluate(X, y, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Save model
    model.save("models/{}".format(NAME))

if __name__ == '__main__':
    main()
