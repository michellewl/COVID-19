from Data_class import DataFrameProcessor, PyTorchDataset
from os import path

data_folder = path.join(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))), "data")
model_folder = path.dirname(path.dirname(path.realpath(__file__)))


textfile = open(path.join(data_folder, "borough_matches.txt"), "r")
boroughs = textfile.read().split("\n")


dfp = DataFrameProcessor(data_folder=data_folder,
                         model_folder=model_folder,
                         boroughs=boroughs,
                         boroughs_descriptor="all_boroughs",
                         species="NO2",
                         training_window=7,
                         quantile_step=0.25)

dfp.create_input_output_arrays(val_size=0.2, test_months=1)

