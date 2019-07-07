import os
from LSTM_Model import LSTM_Model
from TrainingProcess import TrainingProcess
import configparser


class Main(object):
    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.num_epochs = self.config.getint('DEFAULT', 'NUM_EPOCHS')
        self.train_size_per = self.config.getfloat('DEFAULT',
                                                   'TRAIN_SIZE_PERCENT')
        self.num_features = self.config.getint('DEFAULT', 'NUM_FEATURES')
        self.actual_features = self.config.getint('DEFAULT', 'ACTUAL_FEATURES')
        self.activation_function = self.config['DEFAULT'][
            'ACTIVATION_FUNCTION']
        self.loss_function = self.config['DEFAULT']['LOSS_FUNCTION']
        self.metrics = self.config['DEFAULT']['METRICS']
        self.batch_size_train = self.config.getint('DEFAULT',
                                                   'BATCH_SIZE_TRAIN')
        self.batch_size_test = self.config.getint('DEFAULT', 'BATCH_SIZE_TEST')

        self.cur_path = os.path.dirname(__file__)
        self.outputDir = os.path.relpath('../resources/model_resources',
                                         self.cur_path)
        self.datasetDir = os.path.relpath('../data/dataset', self.cur_path)

    def launch(self):
        csv_file = self.datasetDir + \
            '/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

        model = LSTM_Model(self.actual_features, self.num_features, 1,
                           self.activation_function, self.loss_function,
                           self.metrics, self.batch_size_train).create_model()

        tp = TrainingProcess(csv_file, self.num_features, self.train_size_per,
                             self.batch_size_train, self.batch_size_test,
                             self.num_epochs, self.outputDir)
        dataset = tp.read_data_from_csv()

        X_train, Y_train, X_test, Y_test = tp.preprocess(dataset)

        history = tp.train(X_train, Y_train, model)
        tp.evaluate(X_test, Y_test)


if __name__ == '__main__':
    main = Main()
    main.launch()
