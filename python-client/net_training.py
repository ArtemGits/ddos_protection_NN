import json
import optparse
import os

import matplotlib
import numpy as np
import pandas
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from matplotlib.pyplot import figure
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import class_weight

matplotlib.use('agg')

figure(num=None, figsize=(80, 60), dpi=300, facecolor='w', edgecolor='k')




num_features = 83
batch_size_train = 1000  # 1000 1 100
batch_size_test = 1000  # 128 1 64
num_epochs = 10
train_size_per = 0.7

cur_path = os.path.dirname(__file__)
outputDir = os.path.relpath('../resources/model_resources', cur_path)
word_dict_file = os.path.relpath(
    '../resources/dictionary/word-dictionary.json', cur_path)
datasetDir = os.path.relpath('../data/dataset', cur_path)
"""
Create model
"""


def create_model(batch_size):
    model = Sequential()
    model.add(
        LSTM(41, input_shape=(num_features - 1, 1), return_sequences=True))
    model.add(LSTM(20))
    model.add(Dense(units=1, activation='sigmoid'))

    # choose optimizer and loss function
    opt = optimizers.Adam(lr=0.001)

    # compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


"""
Step 1 : Load data
"""


def read_data_from_csv(csv_file):
    dataframe = pandas.read_csv(csv_file)
    dataframe.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    #dataframe.groupby([' Label']).size().plot(kind='bar',stacked=True)
    #sns.heatmap(dataframe.corr(), annot=True)

    #plt.savefig("heatmap.png")

    dataframe.set_value(dataframe[' Label'] == 'BENIGN', [' Label'], 0)
    dataframe.set_value(dataframe[' Label'] == 'DDoS', [' Label'], 1)
    dataframe = dataframe.drop(
        dataframe[(dataframe[' Flow Packets/s'] == 'Infinity') |
                  (dataframe[' Flow Packets/s'] == 'NaN')].index)
    dataframe = dataframe.drop(
        dataframe[(dataframe['Flow Bytes/s'] == 'Infinity') |
                  (dataframe['Flow Bytes/s'] == 'NaN')].index)
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    dataframe = dataframe.dropna()

    dataset = dataframe.values
    # print(dataset)

    # np.save("data/{}.npy".format(os.path.basename(csv_file)), dataset)

    return dataset


def read_data_from_np(npy_file):
    dataset = np.load(npy_file)
    return dataset


"""
Step 2: Preprocess dataset
"""

# TODO try to compile model without word dictionary


def preprocess(dataset):

    print("\nDataset shape: {}".format(dataset.shape))

    #X = dataset[:, :num_features]
    Y = dataset[:, num_features]
    flow_id = np.array(dataset[:, 0]).reshape(-1, 1)
    source_ip = np.array(dataset[:, 1]).reshape(-1, 1)
    destination_ip = np.array(dataset[:, 3]).reshape(-1, 1)
    timestamp = np.array(dataset[:, 6]).reshape(-1, 1)
    # X_ft3 = np.array(dataset[:,2]).reshape(-1, 1)
    X_str = np.concatenate((flow_id, source_ip, destination_ip, timestamp),
                           axis=1)
    # print(X_str)
    """ Vectorize a text corpus, by turning                 each text
    into either a              sequence of
    integers """
    tokenizer = Tokenizer(filters='\t\n', char_level=True, lower=False)
    tokenizer.fit_on_texts(X_str)
    print(os.path.dirname(word_dict_file))
    # Extract and save word dictionary
    if not os.path.exists(os.path.dirname(word_dict_file)):
        os.makedirs(os.path.dirname(word_dict_file))
    with open(word_dict_file, 'w') as outfile:
        json.dump(tokenizer.word_index, outfile, ensure_ascii=False)
        # Transform all text to a sequence of integers
        X_str = tokenizer.texts_to_sequences(X_str)

    X_processed = np.concatenate(
        (np.array(dataset[:, 2]).reshape(-1, 1).astype('float32'), X_str,
         (dataset[:, 4:5]).astype('float32'),
         (dataset[:, 7:num_features]).astype('float32')),
        axis=1)

    print("Features shape: {}".format(X_processed.shape))

    print(Y)

    # Divide to train dataset
    train_size = int(len(dataset) * train_size_per)
    X_train = X_processed[0:train_size]
    Y_train = Y[0:train_size]
    # and test dataset
    X_test = X_processed[train_size:len(X_processed)]
    print("----------------------------------------------------")
    print(X_train)
    Y_test = Y[train_size:len(Y)]

    return X_train, Y_train, X_test, Y_test


"""
Step 2: Train classifier
"""


def train(X_train, Y_train):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #model = create_model(batch_size_train)

    #print(model.summary())
    #print(X_train.shape)

    model = ExtraTreesClassifier()

    # Checkpoint TODO remove checkpoints
    filepath = outputDir + "/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y_train),
                                                      Y_train)
    # Train the model
    model_history = model.fit(X_train, Y_train)

    # display the relative importance of each attribute
    print(model.feature_importances_)

    #print(model_history.history.keys())
    # summarize history for accuracy
    #plt.plot(model_history.history['acc'])
    #plt.plot(model_history.history['val_acc'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig("acc")
    #plt.close()
    # summarize history for loss
    #plt.plot(model_history.history['loss'])
    #plt.plot(model_history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig("loss")
    #plt.close()
    # Save model
    weight_file = '{}/lstm_weights.h5'.format(outputDir)
    model_file = '{}/lstm_model.h5'.format(outputDir)
    model.save_weights(weight_file)
    model.save(model_file)

    return model_history, model, weight_file


"""
Step 3: Evaluate model
"""


def evaluate(X_test, Y_test):
    model = load_model(outputDir + '/lstm_model.h5')
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Evaluate
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size_test)

    print("\nLoss: {}".format(score))
    print("Accuracy: {:0.2f}%".format(acc * 100))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f',
                      '--file',
                      action="store",
                      dest="file",
                      help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = datasetDir + \
            '/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

dataset = read_data_from_csv(csv_file)
X_train, Y_train, X_test, Y_test = preprocess(dataset)

model_history, model, weight_file = train(X_train, Y_train)
#evaluate(X_test, Y_test)
