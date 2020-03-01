import ast
from utiles import find_time_interval, prepare
from networkop import train, test
import click

@click.command()
@click.argument('command', required=1)
@click.option("--in", "-i", "input_path", required=True, help="Path to csv files to be processed.")
@click.option("--out", "-o", "output_path", required=True, default="info.txt", help="Path to text file to store the short info.")
@click.option("--min_size", "-min_s", "min_time_interval_size", required=True, default=7, help="Minimum time interval size.")
@click.option("--min_day", "-min_d", "min_number_of_distinct_days", required=True, default=800, help="Minimum number of distinct days.")
@click.option("--start_date", "-sdate", "start_date_inp", required=True, default="01.01.2015", help="Start date to process the input files '01.01.2015' e.g.")
@click.option("--ti", "-t", "tiflag", required=True, default=1, help="Flag for the additional technical indicators")

def main(command,input_path,output_path,min_time_interval_size,min_number_of_distinct_days,start_date_inp,tiflag):
	""" Process the input files in the given directory
		Train Models
		Make future predictions

		COMMAND: prepare|train|predict|overall

		Sample run command:

		python3 run.py predict -i data -o info.txt -min_s 7 -min_d 800 -sdate '01.01.2015' -t 1
	"""
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
