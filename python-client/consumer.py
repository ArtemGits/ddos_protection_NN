"""Consume data from kafka to make prediction"""

from kafka import KafkaConsumer
from keras.models import load_model
from keras import optimizers
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense


cur_path = os.path.dirname(__file__)
outputDir = os.path.relpath('../resources/model_resources', cur_path)
blackList = os.path.relpath(
    '../resources/black_list/black_list.txt', cur_path)


def model_load():
    num_features = 83

    model = load_model(outputDir+'/lstm_model.h5')

    old_weights = model.get_weights()

    # re-define model
    new_model = Sequential()

    new_model.add(LSTM(82, input_shape=(
        num_features-1, 1), return_sequences=True))
    new_model.add(LSTM(82))
    new_model.add(Dense(units=1, activation='sigmoid'))

    opt = optimizers.Adam(lr=0.001)

    new_model.set_weights(old_weights)

    new_model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    return new_model


def make_prediction(model, dataset):
    print("---------------")

    flow_id = np.array(dataset[:, 0]).reshape(-1, 1)
    source_ip = np.array(dataset[:, 1]).reshape(-1, 1)
    destination_ip = np.array(dataset[:, 3]).reshape(-1, 1)
    timestamp = np.array(dataset[:, 6]).reshape(-1, 1)
    X_str = np.concatenate(
        (flow_id, source_ip, destination_ip, timestamp), axis=1)

    tk = Tokenizer(filters='\t\n', char_level=True, lower=False)
    tk.fit_on_texts(X_str)
    X_str = tk.texts_to_sequences(X_str)

    X_processed = np.concatenate(
        (np.array(dataset[:, 2]).reshape(-1, 1).astype('float32'),
         X_str,
         (dataset[:, 4:5]).astype('float32'),
         (dataset[:, 7:83]).astype('float32')
         ), axis=1)

    X_data = np.reshape(
        X_processed, (X_processed.shape[0], X_processed.shape[1], 1))
    classes = model.predict(X_data, batch_size=1)
    classes = classes.reshape(-1)
    dataset[..., 83] = classes

    black_list = list(set([x[0] for x in dataset[:, [1, 83]] if x[1] >= .5]))
    print(black_list)
    with open(blackList, 'w') as f:
        for ip in black_list:
            f.write("%s\n" % ip)


def kafka_setup():
    # To consume latest messages and auto-commit offsets
    consumer = KafkaConsumer('test-topic',
                             group_id='test-consumer',
                             bootstrap_servers=['kafka:9092'])
    model = model_load()
    for message in consumer:
        make_prediction(model, pickle.loads(message.value))

    # consume earliest available messages, don't commit offsets
    KafkaConsumer(auto_offset_reset='earliest', enable_auto_commit=False)

    # StopIteration if no message after 1sec
    KafkaConsumer(consumer_timeout_ms=1000)


kafka_setup()
