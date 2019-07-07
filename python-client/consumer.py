"""Consume data from kafka to make prediction"""

import configparser
import os
import pickle

import numpy as np
from kafka import KafkaConsumer

from LSTM_Model import LSTM_Model


class Consumer(object):
    def __init__(self):
        self.cur_path = os.path.dirname(__file__)
        self.outputDir = os.path.relpath('../resources/model_resources',
                                         self.cur_path)
        self.blackList = os.path.relpath(
            '../resources/black_list/black_list.txt', self.cur_path)

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.actual_features = self.config.getint('DEFAULT', 'ACTUAL_FEATURES')
        self.activation_function = self.config['DEFAULT'][
            'ACTIVATION_FUNCTION']

        self.number_features = self.config.getint('DEFAULT', 'NUM_FEATURES')

        self.loss_function = self.config['DEFAULT']['LOSS_FUNCTION']
        self.metrics = self.config['DEFAULT']['METRICS']
        self.batch_size_train = self.config.getint('DEFAULT',
                                                   'BATCH_SIZE_TRAIN')

    def model_load(self):
        new_model = LSTM_Model(self.actual_features, self.actual_features, 1,
                               self.activation_function, self.loss_function,
                               self.metrics,
                               self.batch_size_train).create_model()

        return new_model

    def make_prediction(self, model, dataframe):
        print("---------------")

        dataframe['Label'] = np.where(dataframe['Label'] == 'No Label', -1,
                                      dataframe['Label'])

        dataset = dataframe.sample(frac=1).values

        X_processed = np.delete(dataset, [0, 1, 3, 6], 1).astype('float32')

        X_data = np.reshape(X_processed,
                            (X_processed.shape[0], X_processed.shape[1], 1))

        classes = model.predict(X_data, batch_size=1)
        classes = classes.reshape(-1)
        dataset[..., self.number_features] = classes

        # TODO function to write in black_list
        self.black_list = list(
            set([
                x[0] for x in dataset[:, [1, self.number_features]]
                if x[1] >= .5
            ]))
        print(self.black_list)
        with open(self.blackList, 'w') as f:
            for ip in self.black_list:
                f.write("%s\n" % ip)

    def kafka_setup(self):
        # To consume latest messages and auto-commit offsets
        consumer = KafkaConsumer('test-topic',
                                 group_id='test-consumer',
                                 bootstrap_servers=['kafka:9092'])
        model = self.model_load()
        for message in consumer:
            self.make_prediction(model, pickle.loads(message.value))

        # consume earliest available messages, don't commit offsets
        KafkaConsumer(auto_offset_reset='earliest', enable_auto_commit=False)

        # StopIteration if no message after 1sec
        KafkaConsumer(consumer_timeout_ms=1000)


if __name__ == '__main__':
    consumer = Consumer()
    consumer.kafka_setup()
