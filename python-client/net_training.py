"""
Data used for this script: https://github.com/defcom17/NSL_KDD
"""

import sys
import os
import json
import pandas
import numpy as np
import optparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier


from keras import optimizers
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback
from sklearn import preprocessing
from sklearn.utils import class_weight

#import matlotlib.pyplot as plt
#import seaborn as sns; sns.set(style="ticks", color_codes=True)
#from mpl_toolkits.mplot3d import Axes3D


#num_features = 82
num_features = 83
#num_classes = 2
batch_size_train = 1000  # 1000 1 100
batch_size_test = 1000  # 128 1 64
num_epochs = 1
train_size_per = 0.7


outputDir = 'output'


""" 
Modfy model
"""


def create_model(batch_size):
    model = Sequential()
    # model.add(Dense(units=100, input_dim=X_train.shape[1]))
    model.add(LSTM(82, input_shape=(num_features-1, 1), return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(LSTM(82))
    #model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.5))
    #model.add(LSTM(32, recurrent_dropout=0.15, return_sequences=True))
    #model.add(LSTM(16, recurrent_dropout=0.2, return_sequences=True))
    #model.add(LSTM(4, recurrent_dropout=0.3))
    #model.add(LSTM(32, recurrent_dropout=0.1, return_sequences=True))
    #model.add(LSTM(8, recurrent_dropout=0.1))
    #model.add(LSTM(64, recurrent_dropout=0.1, return_sequences=True))
    #model.add(LSTM(64, recurrent_dropout=0.1, return_sequences=True))
    #model.add(LSTM(64, recurrent_dropout=0.1))
    # model.add(LSTM(, recurrent_dropout=0.1))
    # model.add(BatchNormalization())
    #model.add(Dense(units=150, activation='relu'))
    # model.add(Dense(units=50, activation='relu')
    model.add(Dense(units=1, activation='sigmoid'))

    # choose optimizer and loss function
    #opt = optimizers.SGD(lr=0.001)
    opt = optimizers.Adam(lr=0.001)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    # compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer=sgd)
    # model.compile(loss='categorical_crossentropy',
    #              optimizer='rmsprop',
    #              metrics=['accuracy'])
    return model


"""
Step 1 : Load data
"""


def read_data_from_csv(csv_file):
    dataframe = pandas.read_csv(csv_file)
    dataframe.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    dataframe.set_value(dataframe[' Label'] == 'BENIGN', [' Label'], 0)
    dataframe.set_value(dataframe[' Label'] == 'DDoS', [' Label'], 1)
    # plot_model(dataframe)
    dataframe = dataframe.drop(
        dataframe[(dataframe[' Flow Packets/s'] == 'Infinity') |
                  (dataframe[' Flow Packets/s'] == 'NaN')].index)
    dataframe = dataframe.drop(
        dataframe[(dataframe['Flow Bytes/s'] == 'Infinity') |
                  (dataframe['Flow Bytes/s'] == 'NaN')].index)
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    dataframe = dataframe.dropna()

    #min_max_scaler = preprocessing.MinMaxScaler()
    # column_names_to_not_normalize = [
    #    'Flow ID',  ' Source IP', ' Destination IP', ' Timestamp']
    # column_names_to_normalize = [x for x in list(
    #    dataframe) if x not in column_names_to_not_normalize]
    #x = dataframe[column_names_to_normalize]
    #x = x.apply(pandas.to_numeric, errors='coerce')
    # print(x.info())
    #x = x.values
    # print("****************************************************")
    # print(x)

    #x_scaled = min_max_scaler.fit_transform(x)
    # df_temp = pandas.DataFrame(
    #    x_scaled, columns=column_names_to_normalize, index=dataframe.index)
    #dataframe[column_names_to_normalize] = df_temp

    dataset = dataframe.values
    # print(dataframe.info())
    print(dataset)
    #sns_plot = sns.boxplot(data=dataframe)
    #fig = sns_plot.get_figure()
    #fig.set_size_inches(200, 30)
    # fig.savefig("output.png")

    np.save("data/{}.npy".format(os.path.basename(csv_file)), dataset)

    return dataset


def read_data_from_np(npy_file):
    dataset = np.load(npy_file)
    return dataset


"""
Step 2: Preprocess dataset
"""


def preprocess(dataset):
    print("\nDataset shape: {}".format(dataset.shape))

    X = dataset[:, :num_features]
    Y = dataset[:, num_features]
    flow_id = np.array(dataset[:, 0]).reshape(-1, 1)
    source_ip = np.array(dataset[:, 1]).reshape(-1, 1)
    destination_ip = np.array(dataset[:, 3]).reshape(-1, 1)
    timestamp = np.array(dataset[:, 6]).reshape(-1, 1)
    # X_ft3 = np.array(dataset[:,2]).reshape(-1, 1)
    X_str = np.concatenate(
        (flow_id, source_ip, destination_ip, timestamp), axis=1)
    # print(X_str)
    # Vectorize a text corpus, by turning each text into either a sequence of integers
    tokenizer = Tokenizer(filters='\t\n', char_level=True, lower=False)
    tokenizer.fit_on_texts(X_str)
    # Extract and save word dictionary
    word_dict_file = 'build/word-dictionary.json'
    if not os.path.exists(os.path.dirname(word_dict_file)):
        os.makedirs(os.path.dirname(word_dict_file))
    with open(word_dict_file, 'w') as outfile:
        json.dump(tokenizer.word_index, outfile, ensure_ascii=False)
    # Transform all text to a sequence of integers
    #num_words = len(tokenizer.word_index)+1
    X_str = tokenizer.texts_to_sequences(X_str)

    X_processed = np.concatenate(
        (np.array(dataset[:, 2]).reshape(-1, 1).astype('float32'),
         X_str,
         (dataset[:, 4:5]).astype('float32'),
         (dataset[:, 7:num_features]).astype('float32')
         ), axis=1)

    print("Features shape: {}".format(X_processed.shape))

    #Y = to_categorical(Y, num_classes=num_classes)
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

    model = create_model(batch_size_train)
    print(model.summary())
    print(X_train.shape)
    # Checkpoint
    filepath = outputDir+"/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y_train),
                                                      Y_train)
    # Train the model
    model_history = model.fit(X_train, Y_train, class_weight=class_weights, validation_split=0.33,
                              epochs=num_epochs, batch_size=batch_size_train, callbacks=[checkpoint])
    # model_history = model.fit(X_train, Y_train, epochs=num_epochs, callbacks=[plot_losses], batch_size=batch_size_train)

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
    model = load_model(outputDir+'/lstm_model.h5')
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Evaluate
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size_test)
    print("\nLoss: {}".format(score))
    print("Accuracy: {:0.2f}%".format(acc * 100))


# """
# Plot training history
# """
# def plot_model(dataframe):
#     #dataframe.info()
#     #print(dataframe[' Label'].value_counts())
#     #dataframe[' Label'].value_counts().plot(kind='bar', label=' Label')
#     #plt.legend()
#     #plt.title('Distribution of DDoS attacs')
#     #plt.savefig('DDos distribution')

#     #'''build corr_matrix'''
#     #corr_matrix = dataframe.drop([' Source Port', ' Destination Port', ' Protocol'], axis=1).corr()
#     #corr_matrix_plot = sns.heatmap(corr_matrix);
#     #fig = corr_matrix_plot.get_figure()
#     #corr_matrix_plot.get_legend().remove()
#     #fig.set_size_inches(50, 30)
#     #fig.savefig("corr_matrix_quantitative_features.png")
#     features = list(set(dataframe.columns) - set([' Source Port', ' Destination Port', ' Protocol']))
#     #hist_plot = dataframe[features].hist(figsize=(50,30))
#     #plt.savefig('histogram of features distribution')
#     #hist_plot = sns.pairplot(dataframe[features + [' Label']], hue=' Label') TODO
#     #hist_plot.get_figure()
#     '''subplots''' #TODO fix warning with legends
#     fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(30, 20))
#     for idx, feat in  enumerate(features):
#         sns.boxplot(x=' Label', y=feat, data=dataframe, ax=axes[idx // 6, idx % 6])
#         axes[idx // 6, idx % 6].legend()
#         axes[idx // 6, idx % 6].set_xlabel(' Label')
#         axes[idx // 6, idx % 6].set_ylabel(feat);

#     fig.savefig('subplots')


# def feature_analyzer(dataset):
#     X = dataset[:, :num_features]
#     Y = dataset[:, num_features]

#     X_ft1 = np.array(dataset[:,0]).reshape(-1, 1)
#     X_ft3 = np.array(dataset[:,2]).reshape(-1, 1)
#     X_str = np.concatenate((X_ft1, X_ft3), axis=1)

#     # Vectorize a text corpus, by turning each text into either a sequence of integers
#     tokenizer = Tokenizer(filters='\t\n', char_level=True, lower=False)
#     tokenizer.fit_on_texts(X_str)
#     # Extract and save word dictionary
#     word_dict_file = 'build/word-dictionary.json'
#     if not os.path.exists(os.path.dirname(word_dict_file)):
#         os.makedirs(os.path.dirname(word_dict_file))
#     with open(word_dict_file, 'w') as outfile:
#         json.dump(tokenizer.word_index, outfile, ensure_ascii=False)
#     # Transform all text to a sequence of integers
#     #num_words = len(tokenizer.word_index)+1
#     X_str = tokenizer.texts_to_sequences(X_str)

#     X_processed = np.concatenate(
#         ( np.array(dataset[:, 1]).reshape(-1, 1).astype('float32'),
#           X_str,
#           (dataset[:, 3:18]).astype('float32'),
#           (dataset[:, 18:20]).astype('float32'),
#           (dataset[:, 20:num_features]).astype('float32')
#         ), axis=1)


#     columns = [ 'Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets','Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Fwd Packet Length Std','Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes per s', 'Flow Packets per s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min','Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min','Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Mean', 'Packet Length Std', 'Active Mean', 'Active Std', 'Active Max', 'Active Min']
#     #print(Y.tolist())
#     #print(X_processed.shape)
#     np.set_printoptions(suppress=True)
#     model = ExtraTreesClassifier()
#     model.fit(X_processed, Y)
#     feature_importances = model.feature_importances_
#     print(feature_importances)
#     plt.figure(figsize=(20,10))
#     (pandas.Series(feature_importances, index=columns)
#         .plot(kind='barh'))
#     plt.savefig('features_importance')


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store",
                      dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

    #dataset = read_data_from_np('{}.npy'.format(csv_file))
    dataset = read_data_from_csv(csv_file)
    #'''features analyze'''
    # feature_analyzer(dataset)

    X_train, Y_train, X_test, Y_test = preprocess(dataset)
    #print("\nTrain samples: {}".format(X_test[0]))
    #outputDir = 'output/CICID/LSTM_30_04'
    #model = load_model(outputDir+'/lstm_model.h5')
    #opt = optimizers.SGD(lr=0.001)
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # print(X_test.shape[0])
    # print(X_test.shape[1])
    #X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))
    #X_test = np.reshape(X_test[0],(1,35, 1))
    # print(X_test)
    # np.set_printoptions(suppress=True)
    #classes =  model.predict(X_test, verbose=0)
    # print(classes)
    #print("Test samples: {}".format(len(X_test)))

    model_history, model, weight_file = train(X_train, Y_train)
    evaluate(X_test, Y_test)
    # plot_model(model_history)
