import pickle
import random
import sys
from time import sleep

import pandas
from kafka import KafkaClient, KafkaProducer


class DataLoader(object):
    """DataLoader class for loads data to kafka"""

    def __init__(self):
        """__init__ kafka setups"""

        self.KAFKA = KafkaClient('kafka:9092')
        self.PRODUCER = KafkaProducer(bootstrap_servers='kafka:9092',
                                      client_id='test-producer',
                                      max_request_size=30000000)
        self.TOPIC = 'test-topic'

    def read_data_from_csv(self, csv_file):
        """read_data_from_csv func for reads and makes data from csv
        file, which gets from cicflowmeter

        :param csv_file:
        """
        dataframe = pandas.read_csv(csv_file)
        dataframe.loc[dataframe['Label'] == 'No Label', 'Label'] = -1
        try:
            byte_array = pickle.dumps(dataframe)
            self.PRODUCER.send(self.TOPIC, value=byte_array)
        except Exception:
            print(Exception)
        print('pushed:')

        sleep(random.uniform(0.01, 5))


if __name__ == '__main__':
    producer = DataLoader()
    path = sys.argv[1]
    # path = '../data/dataset/test.csv'
    try:
        x_file = open(path)
    except IOError:
        print("Could not read file")
        sys.exit()

        producer.read_data_from_csv(x_file)
