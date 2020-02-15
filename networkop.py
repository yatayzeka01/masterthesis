from numpy import array, column_stack
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
import os
from statistics import mean
from utiles import write_to_file

def train(trainData):
    print("Train Process")
    # Train models using 2 to 6 hours data
    for i in range(2, 7, 1):
        input = []  # numDays x i x 2
        output = []  # numDays

        # Each day array is consisted of 7 rows
        for day in trainData:
            # e.g. prices and volumes array have the size of 2 when i is 2
            # ex open -1 volume -2 sma -3
            # with tiFlag : open, sma, stock, volume, date_time
            prices = array([item[-1] for item in day[:i]])
            volumes = array([item[-2] for item in day[:i]])
            smas = array([item[-3] for item in day[:i]])
            # dayInput has the shape of (2,2) at each iteration for the i value 2
            dayInput = column_stack((prices, volumes, smas)).tolist()
            # append the dayInput to the input list
            input.append(dayInput)

            lastPrice = day[i - 1][-1]
            maxPrice = max([item[-1] for item in day[i:]])
            # heuristic = (maxPrice / lastPrice - 0.97) / (1.03 - 0.97)
            heuristic = maxPrice / lastPrice
            output.append(heuristic)

        # TODO: change input/output range depending on volume/price, muffle, start at 0.5 and up/down, later in day goes to 0
        lstm = LSTM(12, input_shape=(i, 3))
        model = Sequential()
        model.add(lstm)
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

        history = model.fit([input], output, validation_split=0.33, epochs=100, batch_size=72, verbose=0)
        print("\nHistory Metrics: ")
        print("accuracy: ", mean(history.history['accuracy']))
        print("val_accuracy: ", mean(history.history['val_accuracy']))
        print("loss: ", mean(history.history['loss']))
        print("val_loss: ", mean(history.history['val_loss']))

        model_name = "./Models/modelRun2_" + str(i) + ".h5"
        if os.path.exists(model_name):
            os.remove(model_name)
        model.save(model_name)
    return


def test(testData):
    trading_history_file = "trading_history.csv"
    if os.path.exists(trading_history_file):
        os.remove(trading_history_file)

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

        dataGain = dataGain * day[-1][-1] / day[0][-1]


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

            prices = array([item[-1] for item in day[:i]])
            volumes = array([item[-2] for item in day[:i]])
            smas = array([item[-3] for item in day[:i]])

            dayInput = column_stack((prices, volumes, smas)).tolist()
            pred = model.predict([[dayInput]])[0][0]

            # evaluation
            inputeval = []  # numDays x i x 2
            inputeval.append(dayInput)
            outputeval = []  # numDays
            lastPrice = day[i - 1][-1]
            maxPrice = max([item[-1] for item in day[i:]])
            #heuristic = (maxPrice / lastPrice - 0.97) / (1.03 - 0.97)
            heuristic = maxPrice / lastPrice
            outputeval.append(heuristic)

            if pred > 0.58 and not bought:
                # resultseval = model.evaluate([inputeval], outputeval, batch_size=72)
                write_to_file(trading_history_file, 'Intraday buy: {0}'.format(i))
                bought = True
                price = day[i][-1]
                write_to_file(trading_history_file, 'Intraday buy prediction: {0}'.format(pred))
                write_to_file(trading_history_file, 'Intraday buy price: {0}'.format(price))
                write_to_file(trading_history_file, 'Intraday buy moment: {0}'.format(day[i]))
                write_to_file(trading_history_file, "=" * 50)
                buyTime = i
            elif pred < 0.53 and bought:
                write_to_file(trading_history_file, 'Intraday sell: {0}'.format(i))
                bought = False
                modelGain = modelGain * day[i][-1] / price
                write_to_file(trading_history_file, 'Intraday date: {0}'.format(day[0][0]))
                write_to_file(trading_history_file, 'Intraday sell moment: {0}'.format(day[i]))
                write_to_file(trading_history_file, 'Intraday bought at hour {0} for {1}. Sold at hour {2} for {3}.'.format(timechart[buyTime], str(price), timechart[i], str(day[i][-1])))
                write_to_file(trading_history_file, "=" * 50)
        if bought:
            write_to_file(trading_history_file, 'End of day: {0}'.format(i))
            write_to_file(trading_history_file, 'Model Number: {0}'.format(i))
            write_to_file(trading_history_file, 'End of day date: {0}'.format(day[0][0]))
            write_to_file(trading_history_file, 'End of day that moment: {0}'.format(day[i]))
            modelGain = modelGain * day[-1][-1] / price
            write_to_file(trading_history_file, 'End of day bought at hour {0} for {1}. Sold at hour 17:00 for for {2}.'.format(timechart[buyTime], str(price), str(day[i][-1])))
            write_to_file(trading_history_file, "=" * 50)

    print("dataGain: ", dataGain)
    print("modelGain: ", modelGain)
