import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing

SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "BCH-USD"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 5  # how many batches? Try smaller batch if you"re getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
FILE_NAME = "tft_matches.csv"

def preprocess_df(df):
    # this is a list of list that will CONTAIN the sequences and their target value
    all_data = []
    sequential_data = []
    target = []

    time_index = df.iloc[0].name
    for i, row in df.iterrows():
        if time_index == i:
            sequential_data.append([n for n in row[:7]])
            target = [n for n in row[7:]]
        else:
            time_index = i
            # everything but the last element since it will be the attack of the next game
            sequential_data = sequential_data[:-1]
            all_data.append([sequential_data, target])

            # resets the list
            sequential_data = []
            sequential_data.append([n for n in row[:7]])
            target = [n for n in row[7:]]

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


def main():
    main_df = pd.DataFrame() # begin empty

    player_name = []
    target_name = []
    for i in range(7):
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

    ## here, split away some slice of the future data from the main main_df.
    last_20pct = int(len(main_df) * 0.8)
    time_20pct = main_df.iloc[last_20pct].name
    while time_20pct == main_df.iloc[last_20pct].name:
        last_20pct -= 1

    time_20pct = main_df.iloc[last_20pct].name

    validation_main_df = main_df[(main_df.index>=time_20pct)]
    main_df = main_df[(main_df.index<time_20pct)]

    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)

    padded_train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, padding="post")
    padded_validation_x = tf.keras.preprocessing.sequence.pad_sequences(validation_x, padding="post")


    print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    print(f"train_x.shape = {train_x.shape}")
    print(f"validation_x.shape = {validation_x.shape}")
    print(f"padded_train_x.shape = {padded_train_x.shape}")
    print(f"padded_validation_x.shape = {padded_validation_x.shape}")

    model = Sequential()
    model.add(LSTM(128, input_shape=(padded_train_x.shape[1:]), activation="tanh", return_sequences=True))
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

    model.add(Dense(2, activation="softmax"))

    model.summary()


    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    # checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")) # saves only the best ones

    # Train model
    history = model.fit(
        np.asarray(padded_train_x), np.asarray(train_y),
        batch_size=1,
        epochs=EPOCHS,
        validation_data=(np.asarray(padded_validation_x), np.asarray(validation_y))
    )

    # Score model
    score = model.evaluate(validation_x, validation_y, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    # Save model
    model.save("models/{}".format(NAME))

if __name__ == '__main__':
    main()
