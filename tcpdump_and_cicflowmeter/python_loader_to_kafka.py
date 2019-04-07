""" Generate test data to send to Kafka """

import random
from time import sleep
from kafka import KafkaProducer, KafkaClient
import pandas
import pickle


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
    client_id='test-producer'
)
TOPIC = 'test-topic'
# path = 'csv/' + sys.argv[1]
path = '/home/agits/Documents/code/python/docker-kafka-spark-poc/ddos_protection_v0.1/TCPDUMP_and_CICFlowMeter/csv/test.csv'
x_file = open(path)
read_data_from_csv(x_file)
