""" Read test data from Kafka to ensure producer and broker are working """


from kafka import KafkaConsumer
from keras.models import load_model
from keras import optimizers
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import pickle

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def model_load():
    outputDir = 'output'
    num_features = 83
    #num_classes = 2

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
    labels = ['Flow ID',	'Src IP',	'Src Port',	'Dst IP',	'Dst Port',
              'Protocol', 'Timestamp',	'Flow Duration',	'Tot Fwd Pkts',
              'Tot Bwd Pkts',	'TotLen Fwd Pkts',	'TotLen Bwd Pkts',
              'Fwd Pkt Len Max',	'Fwd Pkt Len Min',	'Fwd Pkt Len Mean',
              'Fwd Pkt Len Std',	'Bwd Pkt Len Max',	'Bwd Pkt Len Min',
              'Bwd Pkt Len Mean',	'Bwd Pkt Len Std',	'Flow Byts/s',
              'Flow Pkts/s',	'Flow IAT Mean',	'Flow IAT Std',
              'Flow IAT Max',	'Flow IAT Min',	'Fwd IAT Tot',	'Fwd IAT Mean',
              'Fwd IAT Std',	'Fwd IAT Max',	'Fwd IAT Min',	'Bwd IAT Tot',
              'Bwd IAT Mean',	'Bwd IAT Std',	'Bwd IAT Max',	'Bwd IAT Min',
              'Fwd PSH Flags',	'Bwd PSH Flags',	'Fwd URG Flags',
              'Bwd URG Flags',	'Fwd Header Len',	'Bwd Header Len',
              'Fwd Pkts/s',
              'Bwd Pkts/s',	'Pkt Len Min',	'Pkt Len Max',	'Pkt Len Mean',
              'Pkt Len Std',
              'Pkt Len Var',	'FIN Flag Cnt',	'SYN Flag Cnt',	'RST Flag Cnt',
              'PSH Flag Cnt',	'ACK Flag Cnt', 'URG Flag Cnt',	'CWE Flag Count',
              'ECE Flag Cnt',	'Down/Up Ratio',	'Pkt Size Avg',
              'Fwd Seg Size Avg', 'Bwd Seg Size Avg',	'Fwd Byts/b Avg',
              'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',	'Bwd Byts/b Avg',
              'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',	'Subflow Fwd Pkts',
              'Subflow Fwd Byts', 'Subflow Bwd Pkts',	'Subflow Bwd Byts',
              'Init Fwd Win Byts', 'Init Bwd Win Byts'	'Fwd Act Data Pkts',
              'Fwd Seg Size Min', 'Active Mean',	'Active Std',
              'Active Max',	'Active Min', 'Idle Mean',	'Idle Std',
              'Idle Max',	'Idle Min']

    # print(len(labels))

    # df = pd.DataFrame.from_records(data, columns=labels)

    # dataset = df.values
    # print(dataset)
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
    # print(len(X_data[0]))
    # print(X_data.shape)
    classes = model.predict(X_data, batch_size=1)

    print(classes)


def kafka_setup():
    # To consume latest messages and auto-commit offsets
    consumer = KafkaConsumer('test-topic',
                             group_id='test-consumer',
                             bootstrap_servers=['kafka:9092'])
    model = model_load()
    for message in consumer:
        # print("%s:%d:%d: key=%s value=%s" % (
        # message.topic, message.partition,
        # message.offset, message.key,
        # message.value))
        tmp = pickle.loads(message.value)
        # tmp = tmp.reshape(-1, 84)
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
        # print(tmp)
        make_prediction(model, tmp)

# consume earliest available messages, don't commit offsets
    KafkaConsumer(auto_offset_reset='earliest', enable_auto_commit=False)

# consume json messages
# KafkaConsumer(value_deserializer=lambda m: json.loads(m.decode('ascii')))


# StopIteration if no message after 1sec
    KafkaConsumer(consumer_timeout_ms=1000)


kafka_setup()
