import sys
import ast
from utiles import find_time_interval, prepare
from networkop import train, test

def main():
    # python3 run.py data_ready info.txt 7 800 '01.01.2015' 1
    input_path, output_path, min_time_interval_size, min_number_of_distinct_days, start_date_inp, tiflag = (sys.argv[1:])

    time_interval, valid_stocks_list = find_time_interval(input_path, output_path, int(min_time_interval_size), int(min_number_of_distinct_days), start_date_inp)

    time_interval = ast.literal_eval(time_interval)
    time_interval = [n.strip() for n in time_interval]

    data = prepare(input_path, time_interval, valid_stocks_list, tiflag)
    n = int(0.70 * data.shape[0])
    trainData = data[:n, :, :]
    testData = data[n:, :, :]

    train(trainData)
    test(testData)

if __name__ == "__main__":
    main()
