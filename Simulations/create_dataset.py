import pickle
import os
import lc_model as model
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import parallel_simulation as sim
from sklearn.model_selection import train_test_split
import copy

class SimulationData:
    def __init__(self, data_list, seed=42):
        self.data_list = copy.deepcopy(data_list)
        self.keys = data_list[0]['data'].keys()
        self.num_examples = len(data_list)
        self.timesteps = len(data_list[0]['data']['X']['xdata_masked']) + len(data_list[0]['data']['X']['xdata_unmasked'])
        self.ground_idx_start = None
        self.seed = seed

    def create_data(self, file_path, keys=['X', 'NIR', 'IR', 'submm'], model='triplet', train_size=0.6, val_size=0.3, test_size=0.1):
        
        channels = len(keys)

        if model == 'triplet':    
            file_path = 'data_lib/' + file_path + '_triplet'
            data = np.zeros((self.num_examples, self.timesteps, channels * 2 + 1))
        elif model == 'mogp':
            file_path = 'data_lib/' + file_path + '_mogp'
            data = np.zeros((self.num_examples, self.timesteps, channels * 2))
        

        assert abs(train_size + val_size + test_size - 1) <= 1e-5, 'Train, val, test array sizes must sum to 1'

        for i, data_entry in enumerate(self.data_list):
            for j, key in enumerate(keys):

                # Retrieve unmasked and masked y-values and their corresponding x indices
                ydata_unmasked = np.array(data_entry['data'][key]["ydata_unmasked"])
                xdata_unmasked = np.array(data_entry['data'][key]["xdata_unmasked"], dtype=int)
                ydata_masked = np.array(data_entry['data'][key]["ydata_masked"])
                xdata_masked = np.array(data_entry['data'][key]["xdata_masked"], dtype=int)
                # Fill in unmasked data at the correct time step indices
                data[i, xdata_unmasked, j] = ydata_unmasked
                data[i, xdata_masked, j] = ydata_masked

                # Create a mask: 1 for observed (unmasked), 0 for masked
                mask = np.zeros(self.timesteps)
                mask[xdata_unmasked] = 1  # Mark unmasked data as observed
                data[i, :, channels + j] = mask  # Fill the mask array


            if model == 'triplet':
                # Time progression: assuming it ranges from 0 to 1 over 960 timesteps
                data[i, :, -1] = np.linspace(0, 1, self.timesteps)
            
        train_data, temp_data = train_test_split(data, test_size=1-train_size, random_state=self.seed)
        val_data, test_data = train_test_split(temp_data, test_size=0.1/(1-train_size), random_state=self.seed)

        if model == 'triplet':
            # Save to .npz file
            np.savez(file_path, train=train_data, val=val_data, test=test_data)
        elif model == 'mogp':
            # Save to .npz file
            np.savez(file_path, train=train_data, val=val_data, test=test_data)

    
    def standardize(self):
        for data_entry in self.data_list:
            for key in self.keys:
                ydata_unmasked = np.array(data_entry['data'][key]["ydata_unmasked"])
                data_entry['data'][key]["ydata_unmasked"] = (ydata_unmasked - ydata_unmasked.mean()) / ydata_unmasked.std()


    def NIR_mask(self, data):
        np.random.seed(self.seed)
        
        NIR_ydata = data['NIR']['ydata_unmasked']

        # Define the parameters
        total_time = len(NIR_ydata)  # total number of minutes
        total_observation_window = 4 * 76 + 3 * 40 + np.random.randint(-50, 51)  # Same as sub-mm observation window +/- between 0 and 50 minutes

        # Randomly select a starting point for the window, shifted by ~50 minutes
        if self.ground_idx_start is None:
            # Randomly select a starting point for the window
            max_start_time = total_time - total_observation_window
            self.ground_idx_start = np.random.randint(0, max_start_time + 1)
            random_start_time = self.ground_idx_start
        else:
            random_start_time = self.ground_idx_start + np.random.randint(-50, 51)

        # Ensure the shifted window fits within the total time
        if random_start_time < 0:
            random_start_time = 0
        if random_start_time + total_observation_window > total_time:
            random_start_time = total_time - total_observation_window

        # Generate the time array
        time_array = np.arange(total_time)

        # Initialize lists to hold the indices
        selected_indices = []

        # Generate the indices within the random window
        current_time = random_start_time
        while current_time < random_start_time + total_observation_window:
            observation_period = np.random.randint(1, 11)  # Random observation period between 1 and 10 minutes
            for minute in range(observation_period):
                if current_time + minute < random_start_time + total_observation_window:
                    selected_indices.append(current_time + minute)
            current_time += observation_period + np.random.randint(1, 11)  # Random gap between 1 and 10 minutes

        # Convert selected indices to a numpy array
        selected_indices = np.array(selected_indices)

        # Generate the unmasked indices
        masked_indices = np.setdiff1d(time_array, selected_indices)

        # Here we assume time_xdata is the same as time_array for simplicity
        NIR_xdata_unmasked = time_array[selected_indices]
        NIR_xdata_masked = time_array[masked_indices]
        NIR_ydata_unmasked = NIR_ydata[selected_indices]
        NIR_ydata_masked = NIR_ydata[masked_indices]

        data['NIR']['xdata_unmasked'] = NIR_xdata_unmasked
        data['NIR']['xdata_masked'] = NIR_xdata_masked
        data['NIR']['ydata_unmasked'] = NIR_ydata_unmasked
        data['NIR']['ydata_masked'] = NIR_ydata_masked

    def IR_mask(self, data, key='IR', percentage_removed=0.0):
        np.random.seed(self.seed)

        x_unmasked = data[key]['xdata_unmasked']
        y_unmasked = data[key]['ydata_unmasked']
        x_masked = data[key]['xdata_masked']
        y_masked = data[key]['ydata_masked']

        # Determine the number of points to remove
        num_to_remove = int(percentage_removed * len(x_unmasked))

        # Randomly select indices to remove
        remove_indices = np.random.choice(len(x_unmasked), num_to_remove, replace=False)

        # Get the indices to keep
        keep_indices = np.setdiff1d(np.arange(len(x_unmasked)), remove_indices)

        # Update the unmasked and masked data
        data[key]['xdata_unmasked'] = x_unmasked[keep_indices]
        data[key]['ydata_unmasked'] = y_unmasked[keep_indices]
        data[key]['xdata_masked'] = np.concatenate([x_masked, x_unmasked[remove_indices]])
        data[key]['ydata_masked'] = np.concatenate([y_masked, y_unmasked[remove_indices]])

    def X_mask(self, data, key='X', percentage_removed=0.0):
        np.random.seed(self.seed)

        x_unmasked = data[key]['xdata_unmasked']
        y_unmasked = data[key]['ydata_unmasked']
        x_masked = data[key]['xdata_masked']
        y_masked = data[key]['ydata_masked']

        # Determine the number of points to remove
        num_to_remove = int(percentage_removed * len(x_unmasked))

        # Randomly select indices to remove
        remove_indices = np.random.choice(len(x_unmasked), num_to_remove, replace=False)

        # Get the indices to keep
        keep_indices = np.setdiff1d(np.arange(len(x_unmasked)), remove_indices)

        # Update the unmasked and masked data
        data[key]['xdata_unmasked'] = x_unmasked[keep_indices]
        data[key]['ydata_unmasked'] = y_unmasked[keep_indices]
        data[key]['xdata_masked'] = np.concatenate([x_masked, x_unmasked[remove_indices]])
        data[key]['ydata_masked'] = np.concatenate([y_masked, y_unmasked[remove_indices]]) 

    def submm_mask(self, data):
        np.random.seed(self.seed)
        
        submm_ydata = data['submm']['ydata_unmasked']

        # Define the parameters
        total_time = len(submm_ydata)  # total number of minutes
        num_epochs = 4  # number of observing epochs
        epoch_duration = 76  # duration of each observing epoch in minutes
        gap_between_epochs = 40  # gap between observing epochs in minutes
        observation_period = 7  # duration of each observation period in minutes
        gap_within_epoch = 5  # gap within each observing epoch in minutes
        minutes_between_observations = observation_period + gap_within_epoch

        # Calculate the total duration of the observation window
        total_observation_window = num_epochs * epoch_duration + (num_epochs - 1) * gap_between_epochs

        # Ensure the window can fit within the total time
        if total_observation_window > total_time:
            raise ValueError("The total observation window exceeds the available time.")

        # Generate the time array
        time_array = np.arange(total_time)

        if self.ground_idx_start is None:
            # Randomly select a starting point for the window
            max_start_time = total_time - total_observation_window
            self.ground_idx_start = np.random.randint(0, max_start_time + 1)

        # Initialize lists to hold the indices
        selected_indices = []

        # Generate the indices for each observing epoch within the random window
        for epoch in range(num_epochs):
            start_epoch = self.ground_idx_start + epoch * (epoch_duration + gap_between_epochs)
            end_epoch = start_epoch + epoch_duration
            
            # Generate indices within each epoch
            current_time = start_epoch
            while current_time < end_epoch:
                for minute in range(observation_period):
                    if current_time + minute < end_epoch:
                        selected_indices.append(current_time + minute)
                current_time += minutes_between_observations

        # Convert selected indices to a numpy array
        selected_indices = np.array(selected_indices)

        # Generate the unmasked indices
        masked_indices = np.setdiff1d(time_array, selected_indices)

        # Here we assume time_xdata is the same as time_array for simplicity
        submm_xdata_unmasked = time_array[selected_indices]
        submm_xdata_masked = time_array[masked_indices]
        submm_ydata_unmasked = submm_ydata[selected_indices]
        submm_ydata_masked = submm_ydata[masked_indices]

        data['submm']['xdata_unmasked'] = submm_xdata_unmasked
        data['submm']['xdata_masked'] = submm_xdata_masked
        data['submm']['ydata_unmasked'] = submm_ydata_unmasked
        data['submm']['ydata_masked'] = submm_ydata_masked

    def add_noise(self, data, percentage_removed):
        np.random.seed(self.seed)

        # Randomly remove 10% of the unmasked data and transfer to masked data
        for key in data.keys():
            x_unmasked = data[key]['xdata_unmasked']
            y_unmasked = data[key]['ydata_unmasked']
            x_masked = data[key]['xdata_masked']
            y_masked = data[key]['ydata_masked']

            # Determine the number of points to remove
            num_to_remove = int(percentage_removed * len(x_unmasked))

            # Randomly select indices to remove
            remove_indices = np.random.choice(len(x_unmasked), num_to_remove, replace=False)

            # Get the indices to keep
            keep_indices = np.setdiff1d(np.arange(len(x_unmasked)), remove_indices)

            # Update the unmasked and masked data
            data[key]['xdata_unmasked'] = x_unmasked[keep_indices]
            data[key]['ydata_unmasked'] = y_unmasked[keep_indices]
            data[key]['xdata_masked'] = np.concatenate([x_masked, x_unmasked[remove_indices]])
            data[key]['ydata_masked'] = np.concatenate([y_masked, y_unmasked[remove_indices]])
    
    def mask_data(self, mask=True, percentage_removed=0.3, noise=0.0):

        for data in self.data_list:

            if mask:
                self.submm_mask(data['data'])
                self.NIR_mask(data['data'])
                self.IR_mask(data['data'], percentage_removed=percentage_removed)
                self.X_mask(data['data'], percentage_removed=percentage_removed)
            if noise != 0:
                self.add_noise(data['data'], noise)

    def plot_random_example(self, data_path, fig_path='tmp.png', sim_number=-1, keys=["X", 'NIR', "IR", "submm"], model='triplet'):
        if model == 'triplet':
            data_path = 'data_lib/' + data_path + '_triplet.npz'
        else:
            data_path = 'data_lib/' + data_path + '_mogp.npz'
        # Load the .npz file
        data = np.load(data_path)

        all_data = np.concatenate([data['train'], data['val'], data['test']], axis=0)
        
        # Select a random example
        if sim_number == -1:
            sim_number = np.random.randint(all_data.shape[0])

        example = all_data[sim_number]
        
        timesteps = example.shape[0]
        channels = len(keys)  # X, NIR, IR, submm
        
        # Create a figure with subplots for each channel
        fig, axes = plt.subplots(channels, 1, figsize=(10, 2 * channels), sharex=True, gridspec_kw={'hspace': 0})
        time = np.arange(0, timesteps)

        if channels == 1:
            axes = [axes]
        
        for j, key in enumerate(keys):
            # Retrieve the data and mask for this channel
            ydata = example[:, j]
            mask = example[:, channels + j]
            
            # Plot observed data (where mask is 1)
            axes[j].scatter(time[mask == 1], ydata[mask == 1], s=3, color='#E6C229', label='Observed')
            # Plot masked data (where mask is 0)
            axes[j].scatter(time[mask == 0], ydata[mask == 0], s=3, color='#1B998B', label='Masked')
            
            # Label the subplot
            axes[j].set_ylabel(key)
            axes[j].grid(True)
            axes[j].set_ylim(-3.5, 7.5)
            if j == 0:
                axes[j].legend(loc="upper right")
            
        
        # Set common labels
        axes[-1].set_xlabel("Timesteps")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('test.png')
        plt.show()
