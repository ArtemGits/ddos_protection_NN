""" Read test data from Kafka to ensure producer and broker are working """

import json
from kafka import KafkaConsumer
from keras.models import load_model
from keras import optimizers
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from io import StringIO


def kafka_setup():
	# To consume latest messages and auto-commit offsets
	consumer = KafkaConsumer('test-topic',
	                         group_id='test-consumer',
	                         value_deserializer=lambda m: json.loads(m.decode('ascii')),
	                         bootstrap_servers=['kafka:9092'])
	model = model_load()
	for message in consumer:
	    # message value and key are raw bytes -- decode if necessary!
	    # e.g., for unicode: `message.value.decode('utf-8')`
	   print (" val=%s " % (message.value['frame']))
	   make_prediction(model, message.value['frame'])

	# consume earliest available messages, don't commit offsets
	KafkaConsumer(auto_offset_reset='earliest', enable_auto_commit=False)

	# consume json messages
	#KafkaConsumer(value_deserializer=lambda m: json.loads(m.decode('ascii')))


	# StopIteration if no message after 1sec
	KafkaConsumer(consumer_timeout_ms=1000)

def model_load():
	outputDir = 'output'
	model = load_model(outputDir+'/lstm_model.h5')
	opt = optimizers.SGD(lr=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
	


def make_prediction(model,data):
	#data = [('60950','80','6','22823363','2','6','3','4.242640687','0','0','0','0','0.262888515','0.087629505','22800000','0','22800000','22800000','22800000','22800000','0','22800000','22800000','0','0','0','0','0','0.087629505','0','2','3.464101615','0','0','0','0')]
	labels = ['Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets','Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Fwd Packet Length Std','Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes per s', 'Flow Packets per s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min','Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min','Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Mean', 'Packet Length Std', 'Active Mean', 'Active Std', 'Active Max', 'Active Min']
	#print(type(data))
	# print(len(labels))

	df = pd.DataFrame.from_records(data, columns=labels)

	dataset = df.values
	X_ft1 = np.array(dataset[:,0]).reshape(-1, 1)
	X_ft3 = np.array(dataset[:,2]).reshape(-1, 1)
	X_str = np.concatenate((X_ft1, X_ft3), axis=1)

	tk = Tokenizer(filters='\t\n', char_level=True, lower=False)
	tk.fit_on_texts(X_str)
	X_str = tk.texts_to_sequences(X_str)

	X_processed = np.concatenate(
	        ( np.array(dataset[:, 1]).reshape(-1, 1).astype('float32'), 
	          X_str,
	          (dataset[:, 3:18]).astype('float32'),
	          (dataset[:, 18:20]).astype('float32'),
	          (dataset[:, 20:36:]).astype('float32')
	        ), axis=1)


	X_data = np.reshape(X_processed, (X_processed.shape[0],X_processed.shape[1], 1))
	print(len(X_data[0]))

	classes =  model.predict(X_data, verbose=0)

	print(classes)


kafka_setup()



