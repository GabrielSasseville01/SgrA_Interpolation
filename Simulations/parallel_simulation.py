import pickle
import os
import lc_model as model
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class Simulation:
    def __init__(self, epoch, sampling_rate, VarDict, num_simulations, batch_size=100, checkpoint_file="./checkpoint/checkpoint.pkl", worker_id=None):
        self.epoch = epoch
        self.sampling_rate = sampling_rate
        self.VarDict = VarDict
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.worker_id = worker_id

        # Update checkpoint file with worker ID to ensure uniqueness
        if self.worker_id is not None:
            self.checkpoint_file = f"./checkpoint/checkpoint_worker_{self.worker_id}.pkl"
        else:
            self.checkpoint_file = checkpoint_file
        
        self.start_simulation = 0
        self.batch_id = 0
        
        # Load checkpoint if it exists
        self.load_checkpoint()

    def run_simulation(self, sim_id):
        # Simulation logic (adjust as needed for individual sim_id)
        time = self.epoch * self.sampling_rate

        # Each worker has its own ParticleSystem based on sim_id for unique runs
        run = model.ParticleSystem(f"run.particle", delPoints=1000)

        run.ParamSet(self.VarDict, PickParticle=True)
        run.params_model.update({"noise_22GHz": 0.02})

        generator = model.DataGenerator(run.params_model, time)
        generator.ModelSetup()

        generator.LightCurveData(self.sampling_rate)
        generator.CalculateLightCurves(0, time)

        data = generator.masks(0.2)
        
        return {"simulation_id": sim_id, "data": data}

    def run_and_save_simulations(self):
        # Run the simulations in batches and save the results to files
        while self.start_simulation < self.num_simulations:
            end_simulation = min(self.start_simulation + self.batch_size, self.num_simulations)
            batch_data = [self.run_simulation(sim_id) for sim_id in range(self.start_simulation, end_simulation)]
            
            # Save the batch data
            self.save_data_batch(batch_data, self.batch_id)

            # Update checkpoint and batch ID
            self.batch_id += 1
            self.start_simulation = end_simulation
            self.save_checkpoint()

    def save_data_batch(self, batch_data, batch_id):
        # Use worker ID in the batch file name to prevent overwriting
        if self.worker_id is not None:
            file_name = f"./saved_simulations/data_batch_worker_{self.worker_id}_{batch_id}.pkl"
        else:
            file_name = f"./saved_simulations/data_batch_{batch_id}.pkl"
        
        with open(file_name, "wb") as f:
            pickle.dump(batch_data, f)
        print(f"Saved {file_name}")

    def save_checkpoint(self):
        checkpoint_data = {
            "start_simulation": self.start_simulation,
            "batch_id": self.batch_id
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print(f"Checkpoint saved for worker {self.worker_id}.")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
                self.start_simulation = checkpoint_data["start_simulation"]
                self.batch_id = checkpoint_data["batch_id"]
            print(f"Checkpoint loaded for worker {self.worker_id}.")
        else:
            print(f"No checkpoint found for worker {self.worker_id}. Starting from the beginning.")

    @staticmethod
    def load_data_batch(batch_id, worker_id=None):
        if worker_id is not None:
            file_name = f"./saved_simulations/data_batch_worker_{worker_id}_{batch_id}.pkl"
        else:
            file_name = f"./saved_simulations/data_batch_{batch_id}.pkl"
        
        with open(file_name, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_all_data(num_batches, num_workers):
        all_data = []
        for worker_id in range(num_workers):
            for batch_id in range(num_batches):
                try:
                    batch_data = Simulation.load_data_batch(batch_id, worker_id)
                    all_data.extend(batch_data)
                except FileNotFoundError:
                    print(f"Batch {batch_id} for worker {worker_id} not found.")
        return all_data

    @staticmethod
    def plot_simulation(dataset, sim_number, col=1):
        if col == 1:
            fig, axs = plt.subplots(4, 1, figsize=(12, 18))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            axs = axs.flatten()

        wavelengths = dataset.keys()
        for i, wavelength in enumerate(wavelengths):
            xdata_unmasked = dataset[wavelength]['xdata_unmasked']
            xdata_masked = dataset[wavelength]['xdata_masked']
            ydata_unmasked = dataset[wavelength]['ydata_unmasked']
            ydata_masked = dataset[wavelength]['ydata_masked']

            axs[i].scatter(xdata_unmasked, ydata_unmasked, label='Observed', s=5)
            axs[i].scatter(xdata_masked, ydata_masked, label='Masked', s=5)
            
            axs[i].set_title(f'{wavelength} Data')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Flux')
            axs[i].legend()

        fig.suptitle(f'Simulation {sim_number + 1}', fontsize=16) # +1 because indexing starts at 0
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_all_simulations(all_data):
        for i, dataset in enumerate(all_data):
            Simulation.plot_simulation(dataset, i)

    @staticmethod
    def run_parallel_simulations(epoch, sampling_rate, VarDict, num_simulations, batch_size, num_workers):
        # Divide the total number of simulations among workers
        chunk_size = num_simulations // num_workers
        ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(Simulation.run_batch_for_worker, epoch, sampling_rate, VarDict, start, end, batch_size, worker_id)
                for worker_id, (start, end) in enumerate(ranges)
            ]
            for future in futures:
                future.result()  # Wait for all processes to complete

    def run_batch_for_worker(epoch, sampling_rate, VarDict, start, end, batch_size, worker_id):
        # Create a separate Simulation instance for each worker
        worker_simulation = Simulation(
            epoch=epoch, 
            sampling_rate=sampling_rate, 
            VarDict=VarDict, 
            num_simulations=end - start, 
            batch_size=batch_size, 
            worker_id=worker_id
        )
        
        worker_simulation.run_and_save_simulations()
