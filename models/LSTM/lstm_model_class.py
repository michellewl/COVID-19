from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import torch.nn as nn
from os import path, makedirs
from Data_class import LSTMTorchDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import numpy as np
import pandas as pd

def mape_score(targets, predictions):
    zero_indices = np.where(targets == 0)
    targets_drop_zero = np.delete(targets, zero_indices)
    prediction_drop_zero = np.delete(predictions, zero_indices)
    mape = np.sum(np.abs(targets_drop_zero - prediction_drop_zero)/targets_drop_zero) * 100/len(targets_drop_zero)
    return mape

class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super().__init__()  # runs init for the Module parent class
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)  # LSTM layers
        self.linear = nn.Linear(hidden_layer_size, output_size)  # Linear layers
        # Hidden cell variable contains previous hidden state and previous cell state.
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
        #                     torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # Pass input sequence through the lstm layer, which outputs the layer output, hidden state and cell state
        # at the current time step.

        lstm_out, hidden_state_cell_state = self.lstm(input_seq)#, self.hidden_cell)
        # print(lstm_out.shape, hidden_state_cell_state[0].shape, hidden_state_cell_state[1].shape)
        # Pass the lstm output to the linear layer, which calculates the dot product between
        # the layer input and weight matrix.
        prediction = self.linear(lstm_out[:, -1, :]).reshape(-1)  # We want the most recent hidden state
        # print(prediction.shape)
        return prediction


class LSTMModel():
    def __init__(self, array_folder, hidden_layer_size, batch_size, noise_std=False):
        self.array_folder = array_folder
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.noise_std = noise_std

        model_filename = path.join(self.array_folder, f"model_hl{self.hidden_layer_size}")
        if self.noise_std:
            self.model_filename = f"{model_filename}_augmented" + str(self.noise_std).replace(".", "") + ".tar"
        else:
            self.model_filename = f"{model_filename}.tar"

        self.training_sequences_path = path.join(self.array_folder, "training_sequences.npy")
        self.training_targets_path = path.join(self.array_folder, "training_targets.npy")
        self.validation_sequences_path = path.join(self.array_folder, "validation_sequences.npy")
        self.validation_targets_path = path.join(self.array_folder, "validation_targets.npy")
        self.train_val_sequences_path = path.join(self.array_folder, "train_val_sequences.npy")
        self.train_val_targets_path = path.join(self.array_folder, "train_val_targets.npy")
        self.train_val_dates_path = path.join(self.array_folder, "train_val_dates.npy")
        self.test_dates_path = path.join(self.array_folder, "test_dates.npy")
        self.test_sequences_path = path.join(self.array_folder, "test_sequences.npy")
        self.test_targets_path = path.join(self.array_folder, "test_targets.npy")
        self.x_normaliser_path = path.join(self.array_folder, "x_normaliser.sav")
        self.y_normaliser_path = path.join(self.array_folder, "y_normaliser.sav")

        self.training_dataset = LSTMTorchDataset(self.training_sequences_path, self.training_targets_path, noise_std)
        self.validation_dataset = LSTMTorchDataset(self.validation_sequences_path, self.validation_targets_path)
        self.train_val_dataset = LSTMTorchDataset(self.train_val_sequences_path, self.train_val_targets_path)
        self.test_dataset = LSTMTorchDataset(self.test_sequences_path, self.test_targets_path)

        self.model = LSTMModule(input_size=self.training_dataset.nfeatures(), hidden_layer_size=self.hidden_layer_size)

    def fit(self, num_epochs, learning_rate, batches_per_print=False, epochs_per_print=False):
        torch.manual_seed(5)

        training_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimiser = Adam(self.model.parameters(), lr=learning_rate)

        training_loss_history = []
        validation_loss_history = []
        test_loss_history = []

        print("Begin training...")
        for epoch in range(num_epochs):
            # Training set
            self.model.train()
            loss_sum = 0  # for storing
            running_loss = 0.0  # for printing

            for batch_num, data in enumerate(training_dataloader):
                sequences_training = data["sequences"]
                targets_training = data["targets"]
                optimiser.zero_grad()

                # Run the forward pass
                y_predict = self.model(sequences_training)
                # Compute the loss and gradients
                single_loss = criterion(y_predict, targets_training)
                single_loss.backward()
                # Update the parameters
                optimiser.step()

                # Calculate loss for printing and storing
                running_loss += single_loss.item()
                loss_sum += single_loss.item() * data["targets"].shape[
                    0]  # Account for different batch size with final batch

                # Print the loss after every X batches or after every Y epochs
                if batches_per_print and batch_num % batches_per_print == 0:
                    print(f"epoch: {epoch:3} batch {batch_num} loss: {running_loss / batches_per_print:10.8f}")
                    running_loss = 0.0
            if epochs_per_print and epoch % epochs_per_print == 0:
                print(f"Epoch {epoch} training loss: {loss_sum / len(self.training_dataset)}")
            training_loss_history.append(loss_sum / len(self.training_dataset))  # Save the training loss after every epoch

            # Do the same for the validation set
            self.model.eval()
            validation_loss_sum = 0
            with torch.no_grad():
                for batch_num, data in enumerate(validation_dataloader):
                    sequences_val = data["sequences"]
                    targets_val = data["targets"]
                    y_predict_validation = self.model(sequences_val)
                    single_loss = criterion(y_predict_validation, targets_val)
                    validation_loss_sum += single_loss.item() * data["targets"].shape[0]
                test_loss_sum = 0
                for batch_num, data in enumerate(test_dataloader):
                    sequences_test = data["sequences"]
                    targets_test = data["targets"]
                    y_predict_test = self.model(sequences_test)
                    single_loss = criterion(y_predict_test, targets_test)
                    test_loss_sum += single_loss.item() * data["targets"].shape[0]
                test_loss_history.append(test_loss_sum / len(self.test_dataset))
            # Store the model with smallest validation loss. Check if the validation loss is the lowest BEFORE
            # saving it to loss history (otherwise it will not be lower than itself)
            if (not validation_loss_history) or validation_loss_sum / len(self.validation_dataset) < min(
                    validation_loss_history):
                best_model = deepcopy(self.model.state_dict())
                best_epoch = epoch
            validation_loss_history.append(
                validation_loss_sum / len(self.validation_dataset))  # Save the val loss every epoch.

            # Save the model every 2 epochs
            if epoch % 2 == 0:
                torch.save({
                    "total_epochs": epoch,
                    "final_state_dict": self.model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "training_loss_history": training_loss_history,
                    "best_state_dict": best_model,
                    "best_epoch": best_epoch,
                    "validation_loss_history": validation_loss_history,
                    "test_loss_history": test_loss_history
                }, self.model_filename)

        print(f"Finished training for {num_epochs} epochs.")

    def evaluate(self, training_window, disease, use_overfit_model=False, show_plot=False, plot_title=True, plot_metrics=True):
        train_val_dataloader = DataLoader(self.train_val_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Load the model
        checkpoint = torch.load(path.abspath(self.model_filename))

        if use_overfit_model:
            self.model.load_state_dict(checkpoint["final_state_dict"])
            epoch = checkpoint["total_epochs"]
        else:
            self.model.load_state_dict(checkpoint["best_state_dict"])
            epoch = checkpoint["best_epoch"]

        # Model evaluation
        self.model.eval()

        # Make predictions on training set
        training_targets = []
        training_prediction = []

        with torch.no_grad():
            for batch_num, data in enumerate(train_val_dataloader):
                sequences = data["sequences"]
                targets = data["targets"]

                outputs = self.model(sequences)

                training_targets.append(targets.detach().numpy())
                training_prediction.append(outputs.detach().numpy())

        # Load the normaliser and un-normalise the predictions and targets
        y_normaliser = joblib.load(path.abspath(self.y_normaliser_path))
        training_targets = y_normaliser.inverse_transform(np.concatenate(training_targets, axis=None))
        training_prediction = y_normaliser.inverse_transform(np.concatenate(training_prediction, axis=None))
        print(f"Train targets {training_targets.shape}, Train predict {training_prediction.shape}")

        # Compute the performance metrics
        train_rsq = r2_score(training_targets, training_prediction)
        train_mse = mean_squared_error(training_targets, training_prediction)
        train_mape = mape_score(training_targets, training_prediction)
        print(f"Train R sq {train_rsq}\nTrain MSE {train_mse}\nTrain MAPE {train_mape}")

        # Make predictions on test set
        test_targets = []
        test_prediction = []

        with torch.no_grad():
            for batch_num, data in enumerate(test_dataloader):
                sequences = data["sequences"]
                targets = data["targets"]

                outputs = self.model(sequences)

                test_targets.append(targets.detach().numpy())
                test_prediction.append(outputs.detach().numpy())

        # Un-normalise the test set predictions and targets
        test_targets = y_normaliser.inverse_transform(np.concatenate(test_targets, axis=None))
        test_prediction = y_normaliser.inverse_transform(np.concatenate(test_prediction, axis=None))
        print(f"\nTest targets {test_targets.shape}, Test predict {test_prediction.shape}")

        # Compute the performance metrics
        test_rsq = r2_score(test_targets, test_prediction)
        test_mse = mean_squared_error(test_targets, test_prediction)
        test_mape = mape_score(test_targets, test_prediction)
        print(f"Test R sq {test_rsq}\nTest MSE {test_mse}\nTest MAPE {test_mape}")

        # Make dataframes for the train and test set predictions and targets
        training_dates_boroughs = np.load(path.abspath(self.train_val_dates_path), allow_pickle=True)
        test_dates_boroughs = np.load(path.abspath(self.test_dates_path), allow_pickle=True)

        training_df, test_df = pd.DataFrame(training_dates_boroughs, columns=["date", "borough"]).set_index(
            "date"), pd.DataFrame(test_dates_boroughs, columns=["date", "borough"]).set_index("date")
        training_df["target"], training_df["prediction"] = training_targets, training_prediction
        test_df["target"], test_df["prediction"] = test_targets, test_prediction
        training_df.index, test_df.index = pd.to_datetime(training_df.index), pd.to_datetime(test_df.index)
        print(f"\nTraining dataframe {training_df.shape}, Test dataframe {test_df.shape}")
        boroughs = test_df["borough"].unique()
        print(f"{len(boroughs)} boroughs in test set")

        # Make plots

        # Define the save folder for the plots and create if it doesn't exist already
        save_folder = path.join(self.array_folder, "results_plots")
        if not path.exists(save_folder):
            makedirs(save_folder)
        print("Plotting boroughs...")

        # Create plot for each borough that had test data
        for borough in boroughs:
            fig, axs = plt.subplots(2, 1, figsize=(15, 10))

            # Plot training predictions and targets
            axs[0].plot(training_df.loc[training_df["borough"] == borough].index,
                        training_df.loc[training_df["borough"] == borough, "target"], label="observed")
            axs[0].plot(training_df.loc[training_df["borough"] == borough].index,
                        training_df.loc[training_df["borough"] == borough, "prediction"], label="prediction")
            # Give the plot a title and annotations
            axs[0].set_title(f"Training set ({training_df.index.min()} to {training_df.index.max()})")
            axs[0].legend()
            if plot_metrics:
                axs[0].annotate(f"R$^2$ = {train_rsq}  MSE = {train_mse}  MAPE = {train_mape}", xy=(0.05, 0.92), xycoords="axes fraction",
                            fontsize=12)

            # Plot test predictions and targets
            axs[1].plot(test_df.loc[test_df["borough"] == borough].index, test_df.loc[test_df["borough"] == borough, "target"],
                        label="observed")
            axs[1].plot(test_df.loc[test_df["borough"] == borough].index, test_df.loc[test_df["borough"] == borough, "prediction"],
                        label="prediction")
            # Give the plot a title and annotations
            axs[1].set_title(f"Test set ({test_df.index.min()} to {test_df.index.max()})")
            axs[1].legend()
            if plot_metrics:
                axs[1].annotate(f"R$^2$ = {test_rsq}  MSE = {test_mse}  MAPE = {test_mape}", xy=(0.05, 0.92), xycoords="axes fraction",
                            fontsize=12)

            # Set axes labels for both subplots
            for ax in axs.flatten():
                ax.set_xlabel("Date")
                ax.set_ylabel(f"{disease}")

            # Add an overall title for the figure
            if plot_title:
                fig.suptitle(f"LSTM model for {borough}")

            # Add experiment details as annotations in the figure
            plt.figtext(0.1, 0.5, f"{training_window} day training window",
                        fontsize=12)
            plt.figtext(0.1, 0.48, f"LSTM hidden layer size {self.model.hidden_layer_size}", fontsize=12)
            plt.figtext(0.1, 0.46, f"Model learnt at epoch {epoch}", fontsize=12)

            # Add a legend and adjust figure spacing
            fig.subplots_adjust(top=0.5)
            fig.tight_layout(pad=2)

            # Simplify the borough name for saving
            borough = borough.replace("NHS ", "").replace(" ", "_")
            try:
                borough = borough[:borough.index("_(")]
            except:
                pass

            plot_filename = f"{borough}_timeseries_hl{self.hidden_layer_size}"
            if use_overfit_model:
                plot_filename = f"{borough}_timeseries_overfit_hl{self.hidden_layer_size}"
            if self.noise_std:
                plot_filename += f"_augmented{self.noise_std}".replace(".", "")

            # Save the figure
            fig.savefig(path.join(save_folder, plot_filename + ".png"), dpi=fig.dpi)

            if show_plot:
                plt.show()

            # At the end of the loop, close the figure
            plt.close()

        if not path.exists(path.join(self.array_folder, "metrics.txt")):
            metrics_log = open(path.join(self.array_folder, "metrics.txt"), "w")
        else:
            metrics_log = open(path.join(self.array_folder, "metrics.txt"), "a")

        metrics_log.write("METRICS LOG\n"
                          f"Hidden layer size: {self.hidden_layer_size}\n"
                          f"Data augmentation: Gaussian noise std {self.noise_std}\n"
                          f"Train R sq: {train_rsq}\n"
                          f"Train MSE: {train_mse}\n"
                          f"Train MAPE: {train_mape}\n\n"
                          f"Test R sq: {test_rsq}\n"
                          f"Test MSE: {test_mse}\n"
                          f"Test MAPE: {test_mape}\n\n\n")
        metrics_log.close()

    def plot_loss(self, test_loss=True, show_plot=False):
        # Load training data so we can compute the number of features
        training_dataset = LSTMTorchDataset(self.training_sequences_path, self.training_targets_path)

        # Load the model
        checkpoint = torch.load(path.abspath(self.model_filename))

        # Load the required info for plotting losses
        training_losses = checkpoint["training_loss_history"]
        val_losses = checkpoint["validation_loss_history"]
        best_epoch = checkpoint["best_epoch"]
        epochs = checkpoint["total_epochs"]
        test_losses = checkpoint["test_loss_history"]

        # Plot training and validation loss history and annotate the epoch with best validation loss

        # Initiate the plot figure and define the filename for saving
        fig, ax = plt.subplots(figsize=(12, 8))
        save_name = f"loss_history_hl{self.hidden_layer_size}"

        # Plot training and validation loss histories
        ax.plot(range(epochs + 1), training_losses, label="training loss", alpha=0.8)
        ax.plot(range(epochs + 1), val_losses, label="validation loss", alpha=0.8)

        # Plot test loss history if computed; edit the filename for saving
        if test_loss:
            save_name += "_withtest"
            ax.plot(range(epochs + 1), test_losses, label="test loss", alpha=0.8)

        # Edit the filename for saving the plot if data augmentation was used
        if self.noise_std:
            save_name += f"_augmented{self.noise_std}".replace(".", "")

        # Add a point labelling the epoch with lowest validation loss
        ax.scatter(best_epoch, min(val_losses))

        # Add a legend and title, label the plot and axes
        plt.annotate(f"epoch {best_epoch}", (best_epoch * 1.05, min(val_losses)), fontsize="x-large")
        plt.legend(fontsize="x-large")
        plt.xlabel("epoch", fontsize="x-large")
        plt.ylabel("MSE loss", fontsize="x-large")
        # plt.title(f"LSTM training loss", fontsize="x-large")

        if show_plot:
            plt.show()
        fig.tight_layout()

        # Save the plot as a PNG file
        fig.savefig(path.join(self.array_folder, save_name + ".png"), dpi=fig.dpi)

