import configparser
import os
import pickle

import numpy as np
from kafka import KafkaConsumer
from keras.models import load_model

from LSTM_Model import LSTM_Model


class Consumer(object):
    """Consumer class for receives data from kafka and makes prediction"""

    def __init__(self):
        """__init__ method for first initialization from config"""
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
        """model_load func for loading model from file to make prediction"""

        old_weights = load_model(self.outputDir +
                                 '/lstm_model.h5').get_weights()

        lstm_model = LSTM_Model(self.actual_features, self.actual_features, 1,
                                self.activation_function, self.loss_function,
                                self.metrics, self.batch_size_train)

        new_model = lstm_model.create_model()

        new_model.set_weights(old_weights)
        lstm_model.compile_model(new_model)

        return new_model

    def make_prediction(self, model, dataframe):
        """make_prediction func for make prediction

        :param model - numeral network model:
        :param dataframe - data from kafka broker:
        """
        print("---------------")

        dataset = dataframe.sample(frac=1).values

        X_processed = np.delete(dataset, [0, 1, 3, 6], 1).astype('float32')

        X_data = np.reshape(X_processed,
                            (X_processed.shape[0], X_processed.shape[1], 1))

        classes = model.predict(X_data, batch_size=1)
        classes = classes.reshape(-1)
        dataset[..., self.number_features] = classes

        self.check_and_add_to_blacklist(dataset)

    def check_and_add_to_blacklist(self, dataset):
        """check_and_add_to_blacklist func to finds hacker's
        ip from prediction by neural network.

        :param dataset - data after prediction:
        """
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
        """kafka_setup func to setup up message broker for receives data"""
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
