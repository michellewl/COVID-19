import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from os import path, listdir, makedirs
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Define a function for creating sequences from numpy arrays

def create_x_sequences(x_array, num_sequences, tw):
    input_sequences = []
    for i in range(num_sequences):
        train_sequence = x_array[i:i + tw]
        input_sequences.append(train_sequence)
    input_sequences = np.stack(input_sequences, axis=0)
    return input_sequences

# Define a class for the processed dataframes


class DataFrameProcessor():
    def __init__(self, data_folder, model_folder, boroughs, boroughs_descriptor, species, training_window, quantile_step):
        self.laqn_folder = path.join(data_folder, "daily")
        self.covid_filepath = path.join(data_folder, "london_covid_rate.csv")
        self.boroughs = boroughs
        self.species = species
        self.tw = training_window
        self.qs = quantile_step
        print(f"{len(self.boroughs)} boroughs selected.")

        self.aggregation = [f"{int(method * 100)}_quantile" for method in
                            np.round(np.arange(0, 1 + self.qs, self.qs), 2).tolist()]
        self.laqn_filenames = [f"{self.species}_daily_{method}.csv" for method in self.aggregation]
        self.save_folder = path.join(model_folder, "LSTM", boroughs_descriptor,
                                     f"{species}_{len(self.aggregation)}_quantiles", f"{self.tw}-day_tw")

        if not path.exists(self.save_folder):
            makedirs(self.save_folder)

    def create_input_output_arrays(self, val_size, test_months=1):
        # Load the COVID data
        covid_df = pd.read_csv(self.covid_filepath, index_col="date")
        covid_df.index = pd.to_datetime(covid_df.index)

        # Define training & testing dates
        start_date_covid_train = covid_df.index.min()
        end_date_covid_train = covid_df.index.max() - relativedelta(months=test_months, days=1)
        start_date_covid_test = covid_df.index.max() - relativedelta(months=test_months)
        end_date_covid_test = covid_df.index.max()

        start_date_laqn_train = start_date_covid_train - relativedelta(days=self.tw)
        end_date_laqn_train = end_date_covid_train - relativedelta(days=1)
        start_date_laqn_test = start_date_covid_test - relativedelta(days=self.tw)
        end_date_laqn_test = end_date_covid_test - relativedelta(days=1)

        # print(start_date_laqn_train, start_date_covid_train)
        # print(end_date_laqn_train, end_date_covid_train)
        # print(start_date_laqn_test, start_date_covid_test)
        # print(end_date_laqn_test, end_date_covid_test)

        # Initiate lists for arrays
        # All training arrays, including validation set (used for plotting/evaluation)
        train_val_seq_list = []
        train_val_targ_list = []
        # Training arrays excluding validation set
        train_seq_list = []
        train_targ_list = []
        # Validation arrays
        val_seq_list = []
        val_targ_list = []
        # Test set arrays
        test_seq_list = []
        test_targ_list = []
        # Dates and borough names for training and test sets (needed for plotting/evaluation)
        training_meta_list = []
        test_meta_list = []

        print("Processing boroughs...")
        for borough in self.boroughs:

            # Load the split the laqn data
            laqn_training_array_list = []
            laqn_test_array_list = []
            # Each monthly aggregation statistic is in a separate file
            for laqn_file in self.laqn_filenames:
                # Read the dataframe
                laqn_df = pd.read_csv(path.join(self.laqn_folder, laqn_file)).set_index("date")
                laqn_df.index = pd.to_datetime(laqn_df.index)
                # Isolate the training array
                laqn_array = laqn_df.loc[
                    (laqn_df.index >= start_date_laqn_train) & (laqn_df.index <= end_date_laqn_train),
                    borough].values.reshape(-1, 1)
                laqn_training_array_list.append(laqn_array)
                # Isolate the test array
                laqn_array = laqn_df.loc[(laqn_df.index >= start_date_laqn_test) &
                                         (laqn_df.index <= end_date_laqn_test), borough].values.reshape(-1, 1)
                laqn_test_array_list.append(laqn_array)

            # Join the laqn data arrays for training and testing
            x_train = np.concatenate(laqn_training_array_list, axis=1)
            x_test = np.concatenate(laqn_test_array_list, axis=1)

            # Define the disease data arrays for training and testing
            y_train = covid_df.loc[(covid_df.index >= start_date_covid_train) &
                                   (covid_df.index <= end_date_covid_train), borough].values.reshape(-1, 1)
            y_test = covid_df.loc[(covid_df.index >= start_date_covid_test) &
                                  (covid_df.index <= end_date_covid_test), borough].values.reshape(-1, 1)

            # Create the input sequences for the LSTM
            # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
            train_val_inputs = create_x_sequences(x_train, len(y_train), self.tw)
            test_inputs = create_x_sequences(x_test, len(y_test), self.tw)

            # Split the training and validation sets
            training_sequences, validation_sequences, training_targets, validation_targets = train_test_split(
                train_val_inputs, y_train, test_size=val_size, random_state=1)

            # Add all the arrays to their relevant lists (compiled for all boroughs)
            train_seq_list.append(training_sequences)
            train_targ_list.append(training_targets)
            val_seq_list.append(validation_sequences)
            val_targ_list.append(validation_targets)
            test_seq_list.append(test_inputs)
            test_targ_list.append(y_test)
            train_val_seq_list.append(train_val_inputs)
            train_val_targ_list.append(y_train)

            # Determine the date & borough arrays and append to the relevant lists
            training_date_array = covid_df.loc[
                (covid_df.index >= start_date_covid_train) &
                (covid_df.index <= end_date_covid_train), borough].index.map(str).to_numpy().reshape(-1, 1)
            test_date_array = covid_df.loc[
                (covid_df.index >= start_date_covid_test) &
                (covid_df.index <= end_date_covid_test), borough].index.map(str).to_numpy().reshape(-1, 1)
            borough_train_array = np.repeat(np.array([borough]), training_date_array.shape[0]).reshape(-1, 1)
            borough_test_array = np.repeat(np.array([borough]), test_date_array.shape[0]).reshape(-1, 1)
            training_meta_array = np.concatenate((training_date_array, borough_train_array), axis=1)
            test_meta_array = np.concatenate((test_date_array, borough_test_array), axis=1)
            training_meta_list.append(training_meta_array)
            test_meta_list.append(test_meta_array)

        # Concatentate the full arrays from the lists of arrays per borough
        # Training arrays
        training_sequences = np.concatenate(train_seq_list, axis=0)
        training_targets = np.concatenate(train_targ_list, axis=0)
        # Validation arrays
        validation_sequences = np.concatenate(val_seq_list, axis=0)
        validation_targets = np.concatenate(val_targ_list, axis=0)
        # Test arrays
        test_sequences = np.concatenate(test_seq_list, axis=0)
        test_targets = np.concatenate(test_targ_list, axis=0)
        # Training + validation arrays (for plotting/evaluation)
        train_val_sequences = np.concatenate(train_val_seq_list, axis=0)
        train_val_targets = np.concatenate(train_val_targ_list, axis=0)
        # Dates and boroughs arrays (for plotting/evaluation)
        training_dates = np.concatenate(training_meta_list, axis=0)
        test_dates = np.concatenate(test_meta_list, axis=0)

        # Normalise all the arrays

        # Fit the normaliser to training set
        x_normaliser = StandardScaler().fit(train_val_sequences.reshape(-1, train_val_sequences.shape[2]))
        y_normaliser = StandardScaler().fit(train_val_targets)

        # Save to later apply un-normalisation to test sets for plotting/evaluation
        joblib.dump(x_normaliser, path.join(self.save_folder, "x_normaliser.sav"))
        joblib.dump(y_normaliser, path.join(self.save_folder, f"y_normaliser.sav"))

        # Normalise input and output data
        training_sequences = x_normaliser.transform(
            training_sequences.reshape(-1, training_sequences.shape[2])).reshape(training_sequences.shape)
        training_targets = y_normaliser.transform(training_targets)
        validation_sequences = x_normaliser.transform(
            validation_sequences.reshape(-1, validation_sequences.shape[2])).reshape(validation_sequences.shape)
        validation_targets = y_normaliser.transform(validation_targets)
        train_val_sequences = x_normaliser.transform(
            train_val_sequences.reshape(-1, train_val_sequences.shape[2])).reshape(train_val_sequences.shape)
        train_val_targets = y_normaliser.transform(train_val_targets)
        test_sequences = x_normaliser.transform(test_sequences.reshape(-1, test_sequences.shape[2])).reshape(
            test_sequences.shape)
        test_targets = y_normaliser.transform(test_targets)

        print(f"\nDropping NaNs\nTraining {np.isnan(training_sequences).any(axis=(1, 2)).sum()}\n"
              f"Validation {np.isnan(validation_sequences).any(axis=(1, 2)).sum()}\n"
              f"Train/val {np.isnan(train_val_sequences).any(axis=(1, 2)).sum()}\n"
              f"Test {np.isnan(test_sequences).any(axis=(1, 2)).sum()}")

        # Look along dimensions 1 & 2 for NaNs
        training_sequences_dropna = training_sequences[np.logical_not(np.isnan(training_sequences).any(axis=(1, 2)))]
        training_targets_dropna = training_targets[np.logical_not(np.isnan(training_sequences).any(axis=(1, 2)))]
        validation_sequences_dropna = validation_sequences[
            np.logical_not(np.isnan(validation_sequences).any(axis=(1, 2)))]
        validation_targets_dropna = validation_targets[np.logical_not(np.isnan(validation_sequences).any(axis=(1, 2)))]
        train_val_sequences_dropna = train_val_sequences[np.logical_not(np.isnan(train_val_sequences).any(axis=(1, 2)))]
        train_val_targets_dropna = train_val_targets[np.logical_not(np.isnan(train_val_sequences).any(axis=(1, 2)))]
        test_sequences_dropna = test_sequences[np.logical_not(np.isnan(test_sequences).any(axis=(1, 2)))]
        test_targets_dropna = test_targets[np.logical_not(np.isnan(test_sequences).any(axis=(1, 2)))]
        training_dates_dropna = training_dates[np.logical_not(np.isnan(train_val_sequences).any(axis=(1, 2)))]
        test_dates_dropna = test_dates[np.logical_not(np.isnan(test_sequences).any(axis=(1, 2)))]

        print(
            f"\nTraining sequences {training_sequences_dropna.shape} Training targets {training_targets_dropna.shape}")
        print(
            f"Validation sequences {validation_sequences_dropna.shape} Validation targets {validation_targets_dropna.shape}")
        print(
            f"Train/val sequences {train_val_sequences_dropna.shape} Train/val targets {train_val_targets_dropna.shape} Training dates {training_dates_dropna.shape}")
        print(f"Test sequences {test_sequences_dropna.shape} Test targets {test_targets_dropna.shape} Test dates {test_dates_dropna.shape}")

        # Save the arrays
        np.save(path.join(self.save_folder, "training_sequences.npy"), training_sequences_dropna)
        np.save(path.join(self.save_folder, "validation_sequences.npy"), validation_sequences_dropna)
        np.save(path.join(self.save_folder, f"training_targets.npy"), training_targets_dropna)
        np.save(path.join(self.save_folder, f"validation_targets.npy"), validation_targets_dropna)
        np.save(path.join(self.save_folder, "train_val_sequences.npy"), train_val_sequences_dropna)
        np.save(path.join(self.save_folder, f"train_val_targets.npy"), train_val_targets_dropna)

        np.save(path.join(self.save_folder, "test_sequences.npy"), test_sequences_dropna)
        np.save(path.join(self.save_folder, f"test_targets.npy"), test_targets_dropna)

        np.save(path.join(self.save_folder, "train_val_dates.npy"), training_dates_dropna)
        np.save(path.join(self.save_folder, f"test_dates.npy"), test_dates_dropna)

        print("\nSaved npy arrays.")


class PyTorchDataset(Dataset):
    def __init__(self, sequences_path, targets_path, noise_std=False):
        self.inputs = torch.from_numpy(np.load(sequences_path)).float()
        self.targets = torch.from_numpy(np.load(targets_path)).float()
        self.noise_std = noise_std  # Standard deviation of Gaussian noise

    def __len__(self):
        return self.inputs.size()[0]

    def nfeatures(self):
        return self.inputs.size()[-1]

    def sequence_size(self):
        return self.inputs.size()[1]

    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        if self.noise_std:
            noise = torch.randn_like(input)*self.noise_std
            input = input + noise
        return {"sequences": input, "targets": target}
