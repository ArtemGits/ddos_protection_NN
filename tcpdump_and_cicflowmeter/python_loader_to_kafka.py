""" Generate test data to send to Kafka """

import random
from time import sleep
from kafka import KafkaProducer, KafkaClient
import pandas
import pickle
import sys

def read_data_from_csv(csv_file):
    dataframe = pandas.read_csv(csv_file)
    dataset = dataframe.sample(frac=1).values
    print(dataset)
    print(dataset.dtype)
    print(dataset.shape)
    try:
        byte_array = pickle.dumps(dataset)
        PRODUCER.send(TOPIC, value=byte_array)
    except Exception:
        print(Exception)
    print('pushed:')

    sleep(random.uniform(0.01, 5))


"""
Setup PRODUCER
"""

KAFKA = KafkaClient('kafka:9092')
PRODUCER = KafkaProducer(
    bootstrap_servers='kafka:9092',
    client_id='test-producer',
    max_request_size=30000000
)
TOPIC = 'test-topic'
path = 'csv/' + sys.argv[1]
#path = '/home/agits/Documents/code/python/ddos_protection_NN/data/dataset/test.csv'
#path = '/home/agits/Documents/reports/csv/2019-05-09-19:51:14_ISCX.csv'
#path = '/home/agits/Documents/reports/csv/2019-05-09-20:00:14_ISCX.csv'
#path = '/home/agits/Documents/reports/csv/2019-05-09-19:52:14_ISCX.csv'
x_file = open(path)
read_data_from_csv(x_file)
