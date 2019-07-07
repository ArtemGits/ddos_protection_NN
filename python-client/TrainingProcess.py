import pandas
import numpy as np

from keras.models import load_model
from sklearn.utils import class_weight


class TrainingProcess(object):
    def __init__(self, csv_file, num_features, train_size_per,
                 batch_size_train, batch_size_test, num_epochs, outputDir):
        self.csv_file = csv_file
        self.num_features = num_features
        self.train_size_per = train_size_per
        self.batch_size_train = batch_size_train
        self.num_epochs = num_epochs
        self.outputDir = outputDir
        self.batch_size_test = batch_size_test

    def read_data_from_csv(self):
        dataframe = pandas.read_csv(self.csv_file)
        dataframe.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        dataframe[' Label'] = np.where(dataframe[' Label'] == 'DDoS', 1,
                                       dataframe[' Label'])
        dataframe[' Label'] = np.where(dataframe[' Label'] == 'BENIGN', 0,
                                       dataframe[' Label'])

        dataframe = dataframe.drop(
            dataframe[(dataframe[' Flow Packets/s'] == 'Infinity') |
                      (dataframe[' Flow Packets/s'] == 'NaN')].index)
        dataframe = dataframe.drop(
            dataframe[(dataframe['Flow Bytes/s'] == 'Infinity') |
                      (dataframe['Flow Bytes/s'] == 'NaN')].index)
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        dataframe = dataframe.dropna()

        dataset = dataframe.values

        return dataset

    def preprocess(self, dataset):
        print("\nDataset shape: {}".format(dataset.shape))

        Y = dataset[:, self.num_features]

        X_processed = np.delete(dataset, [0, 1, 3, 6], 1).astype('float32')

        # Divide to train dataset
        train_size = int(len(dataset) * self.train_size_per)
        X_train = X_processed[0:train_size]
        Y_train = Y[0:train_size]
        # and test dataset
        X_test = X_processed[train_size:len(X_processed)]
        Y_test = Y[train_size:len(Y)]

        return X_train, Y_train, X_test, Y_test

    def train(self, X_train, Y_train, model):
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(Y_train), Y_train)
        # Train the model
        model_history = model.fit(
            X_train,
            Y_train,
            class_weight=class_weights,
            validation_split=0.33,
            epochs=self.num_epochs,
            batch_size=self.batch_size_train,
        )

        # Save model
        weight_file = '{}/lstm_weights.h5'.format(self.outputDir)
        model_file = '{}/lstm_model.h5'.format(self.outputDir)
        model.save_weights(weight_file)
        model.save(model_file)

        return model_history

    def evaluate(self, X_test, Y_test):
        model = load_model(self.outputDir + '/lstm_model.h5')
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Evaluate
        score, acc = model.evaluate(X_test,
                                    Y_test,
                                    batch_size=self.batch_size_test)

        print("\nLoss: {}".format(score))
        print("Accuracy: {:0.2f}%".format(acc * 100))
