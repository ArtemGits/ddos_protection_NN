from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense


class LSTM_Model(object):
    """LSTM_Model class for creates model, compiles model"""

    def __init__(self, lstm_units, num_features, dense_units, activation, loss,
                 metrics, batch_size):
        """__init__ first initialization

        :param lstm_units: - neurons for lstm layers
        :param num_features: - number of dataset features
        :param dense_units: - neurons of dense layer
        :param activation: - activation function
        :param loss: - loss function
        :param metrics: - metrics to watch traning process
        :param batch_size: - step of iteration while trains model
        """

        self.lstm_units = lstm_units
        self.num_features = num_features
        self.dense_units = dense_units
        self.activation = activation
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        # choose optimizer and loss function
        self.opt = optimizers.Adam(lr=0.001)

    def create_model(self):
        """create_model func for creates model"""

        model = Sequential()
        model.add(
            LSTM(self.lstm_units,
                 input_shape=(self.lstm_units, 1),
                 return_sequences=True))
        model.add(LSTM(self.lstm_units))
        model.add(Dense(units=self.dense_units, activation=self.activation))

        return model

    def compile_model(self, model):
        """compile_model func for compiles the model

        :param model: - neural network model
        """
        # compile the model
        model.compile(loss=self.loss,
                      optimizer=self.opt,
                      metrics=[self.metrics])
