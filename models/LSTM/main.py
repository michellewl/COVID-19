from Data_class import DataFrameProcessor
from lstm_model_class import LSTMModel
from os import path

first_run = False
evaluate_only = True

# ----------------------------------------------------------------------------------------------------------------------
#              Definitions
# ----------------------------------------------------------------------------------------------------------------------

data_folder = path.join(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))), "data")
model_folder = path.dirname(path.dirname(path.realpath(__file__)))

textfile = open(path.join(data_folder, "borough_matches.txt"), "r")
boroughs = textfile.read().split("\n")

boroughs_descriptor = "all_boroughs"
species = "NO2"
training_window = 7
quantile_step = 0.25
# ----------------------------------------------------------------------------------------------------------------------
#         Process data for modelling
# ----------------------------------------------------------------------------------------------------------------------

dfp = DataFrameProcessor(data_folder=data_folder,
                         target_filename="london_covid_daily_numbers.csv",
                         model_folder=model_folder,
                         boroughs=boroughs,
                         boroughs_descriptor=boroughs_descriptor,
                         species=species,
                         training_window=training_window,
                         quantile_step=quantile_step)
if first_run:
    dfp.create_input_output_arrays(val_size=0.2, test_months=1)

# ----------------------------------------------------------------------------------------------------------------------
#              LSTM model
# ----------------------------------------------------------------------------------------------------------------------

model = LSTMModel(
    array_folder=path.join(model_folder, "LSTM", boroughs_descriptor, f"{species}_{int(1/quantile_step)}_quantiles",
                           f"{training_window}-day_tw"),
    hidden_layer_size=4,
    batch_size=32)

if not evaluate_only:
    model.fit(num_epochs=5000,
              learning_rate=0.001,
              epochs_per_print=10)

model.plot_loss(test_loss=True)

model.evaluate(training_window=7,
               disease="COVID-19 daily lab-confirmed cases")