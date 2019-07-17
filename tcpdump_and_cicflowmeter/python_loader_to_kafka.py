""" Generate test data to send to Kafka """

import pickle
import random
import sys
from time import sleep

import pandas
from kafka import KafkaClient, KafkaProducer


class DataLoader(object):
    
    def __init__(self):
        """
        Setup PRODUCER
        """

        self.KAFKA = KafkaClient('kafka:9092')
        self.PRODUCER = KafkaProducer(bootstrap_servers='kafka:9092',
                                      client_id='test-producer',
                                      max_request_size=30000000)
        self.TOPIC = 'test-topic'

    def read_data_from_csv(self, csv_file):
        dataframe = pandas.read_csv(csv_file)
        dataset = dataframe.sample(frac=1).values
        #    print(dataset)
        #    print(dataset.dtype)
        #    print(dataset.shape)
        try:
            byte_array = pickle.dumps(dataset)
            self.PRODUCER.send(self.TOPIC, value=byte_array)
        except Exception:
            print(Exception)
        print('pushed:')

        sleep(random.uniform(0.01, 5))

if __name__ == '__main__':
    producer = DataLoader()
    path = 'csv/' + sys.argv[1]
    #path = '/home/agits/Documents/code/python/ddos_protection_NN/data/dataset/test.csv'
    #path = '/home/agits/Documents/reports/csv/2019-05-09-19:51:14_ISCX.csv'
    #path = '/home/agits/Documents/reports/csv/2019-05-09-20:00:14_ISCX.csv'
    #path = '/home/agits/Documents/reports/csv/2019-05-09-19:52:14_ISCX.csv'
    x_file = open(path)
    producer.read_data_from_csv(x_file)
