import warnings

warnings.filterwarnings('ignore')
from numpy import array, column_stack
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
import os
from statistics import mean
from utiles import write_to_file
from utiles import plot_heatmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def train(trainData, width, height, finalDataCoList):
	print("Train Process")
	finalDataCoDict = {}
	for i in range(len(finalDataCoList)):
		finalDataCoDict[finalDataCoList[i]] = i

	'''
	del finalDataCoDict["date_time"]
	del finalDataCoDict["Stock"]
	finalDataCoList.remove('date_time')
	finalDataCoList.remove('Stock')
	print(finalDataCoList)
	print(finalDataCoDict)
	'''

	if 'date_time' in finalDataCoList:
		finalDataCoList.remove('date_time')
	if 'Stock' in finalDataCoList:
		finalDataCoList.remove('Stock')
	if 'date' in finalDataCoList:
		finalDataCoList.remove('date')
	if 'time' in finalDataCoList:
		finalDataCoList.remove('time')

	if 'date_time' in finalDataCoDict:
		del finalDataCoDict["date_time"]
	if 'Stock' in finalDataCoDict:
		del finalDataCoDict["Stock"]
	if 'date' in finalDataCoDict:
		del finalDataCoDict["date"]
	if 'time' in finalDataCoDict:
		del finalDataCoDict["time"]

	print(finalDataCoList)
	print(finalDataCoDict)

	if not os.path.exists('Models'):
		os.mkdir('Models')
	plt.figure(figsize=(width / 100., height / 100.), dpi=100)
	color = iter(plt.cm.jet(np.linspace(0, 1, 5)))
	# Train models using 2 to 6 hours data
	for i in range(2, 7, 1):
		input = [] # numDays x i x 2
		output = [] # numDays

		# Each day array is consisted of 7 rows
		for day in trainData:
			dictlist = []

			# e.g. prices and volumes array have the size of 2 when i is 2
			# ex open -1 volume -2 sma -3
			# with tiFlagOld : open, sma, stock, volume, date_time
			# with tiFlag : date_time, open, volume, sma, stock
			# prices = array([item[1] for item in day[:i]])
			for iterkey in finalDataCoDict:
				globals()[iterkey] = array([item[finalDataCoDict[iterkey]] for item in day[:i]])
				dictlist.append(iterkey)

			# dayInput has the shape of (2,2) at each iteration for the i value 2
			dayInput = column_stack((globals()[i] for i in dictlist)).tolist()

			# append the dayInput to the input list
			input.append(dayInput)

			lastPrice = day[i - 1][finalDataCoDict['Open']]
			maxPrice = max([item[finalDataCoDict['Open']] for item in day[i:]])
			# heuristic = (maxPrice / lastPrice - 0.97) / (1.03 - 0.97)
			heuristic = maxPrice / lastPrice
			output.append(heuristic)

		# TODO: change input/output range depending on volume/price, muffle, start at 0.5 and up/down, later in day goes to 0
		lstm = LSTM(12, input_shape=(i, len(finalDataCoList)))
		model = Sequential()
		model.add(lstm)
		model.add(Dense(1))
		model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])

		history = model.fit([input], output, validation_split=0.33, epochs=100, batch_size=72, verbose=0)
		print("\nHistory Metrics:")
		print("mean_squared_error: ", mean(history.history['mean_squared_error']))
		print("val_mean_squared_error: ", mean(history.history['val_mean_squared_error']))
		print("loss: ", mean(history.history['loss']))
		print("val_loss: ", mean(history.history['val_loss']))

		new_input = []
		for b in input:
			col_totals = [sum(x) / len(b) for x in zip(*b)]
			new_input.append(col_totals)

		iterasyon = 0
		for o in output:
			new_input[iterasyon].append(o)
			iterasyon = iterasyon + 1

		dictlist.append("Heuristic Price")
		inpout = pd.DataFrame(new_input, columns=dictlist)

		figure_name = "correlation_plot_" + str(i) + ".png"
		plot_heatmap(figure_name, inpout, i, width, height)

		c = next(color)
		plt.subplot(4, 1, 1)
		plt.plot(history.history['mean_squared_error'], c=c, label=i)
		plt.legend(loc="upper left", fontsize=10)
		plt.title('Training Metrics')
		plt.ylabel('mean_squared_error')

		plt.subplot(4, 1, 2)
		plt.plot(history.history['val_mean_squared_error'], c=c, label=i)
		plt.legend(loc="upper left", fontsize=10)
		plt.ylabel('val_mean_squared_error')

		plt.subplot(4, 1, 3)
		plt.plot(history.history['loss'], c=c, label=i)
		plt.legend(loc="upper left", fontsize=10)
		plt.ylabel('loss')

		plt.subplot(4, 1, 4)
		plt.plot(history.history['val_loss'], c=c, label=i)
		plt.legend(loc="upper left", fontsize=10)
		plt.xlabel('Epoch')
		plt.ylabel('val_loss')

		model_name = "./Models/modelRun2_" + str(i) + ".h5"
		if os.path.exists(model_name):
			os.remove(model_name)
		model.save(model_name)

	plt.savefig("train_metrics_plot.png", dpi=600)

	plt.close()
	return

def test(testData, width, height, finalDataCoList):
	finalDataCoDict = {}
	for i in range(len(finalDataCoList)):
		finalDataCoDict[finalDataCoList[i]] = i

	'''
	del finalDataCoDict["date_time"]
	del finalDataCoDict["Stock"]
	finalDataCoList.remove('date_time')
	finalDataCoList.remove('Stock')
	'''

	if 'date_time' in finalDataCoList:
		finalDataCoList.remove('date_time')
	if 'Stock' in finalDataCoList:
		finalDataCoList.remove('Stock')
	if 'date' in finalDataCoList:
		finalDataCoList.remove('date')
	if 'time' in finalDataCoList:
		finalDataCoList.remove('time')

	if 'date_time' in finalDataCoDict:
		del finalDataCoDict["date_time"]
	if 'Stock' in finalDataCoDict:
		del finalDataCoDict["Stock"]
	if 'date' in finalDataCoDict:
		del finalDataCoDict["date"]
	if 'time' in finalDataCoDict:
		del finalDataCoDict["time"]

	print(finalDataCoList)
	print(finalDataCoDict)

	trading_history_file = "trading_history.csv"
	if os.path.exists(trading_history_file):
		os.remove(trading_history_file)
	plot_pred = defaultdict(list)
	plot_heuristic = defaultdict(list)

	print("\nTest Process")
	modelGain = 1
	dataGain = 1

	timechart = {0: "10:00", 1: "11:00", 2: "12:00", 3: "14:00", 4: "15:00", 5: "16:00", 6: "17:00"}

	model2 = load_model("./Models/modelRun2_2.h5")
	model3 = load_model("./Models/modelRun2_3.h5")
	model4 = load_model("./Models/modelRun2_4.h5")
	model5 = load_model("./Models/modelRun2_5.h5")
	model6 = load_model("./Models/modelRun2_6.h5")

	for day in testData:

		dataGain = dataGain * day[-1][finalDataCoDict['Open']] / day[0][finalDataCoDict['Open']]

		bought = False
		price = 0
		buyTime = 0

		for i in range(2, 7, 1):
			if i == 2:
				model = model2
			elif i == 3:
				model = model3
			elif i == 4:
				model = model4
			elif i == 5:
				model = model5
			elif i == 6:
				model = model6

			dictlist = []

			for iterkey in finalDataCoDict:
				globals()[iterkey] = array([item[finalDataCoDict[iterkey]] for item in day[:i]])
				dictlist.append(iterkey)

			dayInput = column_stack((globals()[i] for i in dictlist)).tolist()
			pred = model.predict([[dayInput]])[0][0]

			# evaluation
			inputeval = [] # numDays x i x 2
			inputeval.append(dayInput)
			outputeval = [] # numDays
			lastPrice = day[i - 1][finalDataCoDict['Open']]
			maxPrice = max([item[finalDataCoDict['Open']] for item in day[i:]])
			# heuristic = (maxPrice / lastPrice - 0.97) / (1.03 - 0.97)
			heuristic = maxPrice / lastPrice
			outputeval.append(heuristic)

			plot_pred[i].append(float(pred))
			plot_heuristic[i].append(heuristic)

			if pred > 0.75 and not bought:
				write_to_file(trading_history_file, 'Intraday buy: {0}'.format(i))
				bought = True
				price = day[i][finalDataCoDict['Open']]
				write_to_file(trading_history_file, 'Intraday buy prediction: {0}'.format(pred))
				write_to_file(trading_history_file, 'Intraday buy price: {0}'.format(price))
				write_to_file(trading_history_file, 'Intraday buy moment: {0}'.format(day[i]))
				write_to_file(trading_history_file, "=" * 50)
				buyTime = i
			elif pred < 0.70 and bought:
				write_to_file(trading_history_file, 'Intraday sell: {0}'.format(i))
				bought = False
				modelGain = modelGain * day[i][finalDataCoDict['Open']] / price
				write_to_file(trading_history_file, 'Intraday date: {0}'.format(day[0][0]))
				write_to_file(trading_history_file, 'Intraday sell moment: {0}'.format(day[i]))
				write_to_file(trading_history_file,
							'Intraday bought at hour {0} for {1}. Sold at hour {2} for {3}.'.format(
								timechart[buyTime], str(price), timechart[i], str(day[i][finalDataCoDict['Open']])))
				write_to_file(trading_history_file, "=" * 50)
		if bought:
			write_to_file(trading_history_file, 'End of day: {0}'.format(i))
			write_to_file(trading_history_file, 'Model Number: {0}'.format(i))
			write_to_file(trading_history_file, 'End of day date: {0}'.format(day[0][0]))
			write_to_file(trading_history_file, 'End of day that moment: {0}'.format(day[i]))
			modelGain = modelGain * day[-1][finalDataCoDict['Open']] / price
			write_to_file(trading_history_file,
						'End of day bought at hour {0} for {1}. Sold at hour 17:00 for for {2}.'.format(
							timechart[buyTime], str(price), str(day[i][finalDataCoDict['Open']])))
			write_to_file(trading_history_file, "=" * 50)

	print("dataGain: ", dataGain)
	print("modelGain: ", modelGain)

	fig = plt.figure(figsize=(width / 100., height / 100.), dpi=100)

	plot_position = 1
	for x in range(2, 7):
		plt.subplots_adjust(bottom=0.08, hspace=1, right=0.85, top=0.95)
		outputpredToPlotDF = pd.DataFrame({'prediction': plot_pred[x][::10]})
		outputheuristicToPlotDF = pd.DataFrame({'heuristic': plot_heuristic[x][::10]})
		plt.subplot(5, 1, plot_position)
		plt.plot(outputheuristicToPlotDF, color="blue", label="Model {} Heuristic price".format(x))
		plt.plot(outputpredToPlotDF, color="red", label="Model {} Prediction price".format(x))
		plt.legend(loc="upper left", fontsize=6)
		plt.ylabel('Model {0} value'.format(x))
		plt.title('Test Metrics for Model {}'.format(x), fontdict={'fontsize': 8})
		plt.xlabel('Hours')
		plt.ylabel('Value')
		plot_position += 1
	fig.savefig("test_metrics_plot.png", dpi=600)
	plt.close()
