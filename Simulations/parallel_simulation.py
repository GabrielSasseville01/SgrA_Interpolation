import pickle
import os
import lc_model as model
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class Simulation:
    def __init__(self, epoch, sampling_rate, VarDict, num_simulations, batch_size=100, checkpoint_file="checkpoint.pkl", worker_id=None):
        self.epoch = epoch
        self.sampling_rate = sampling_rate
        self.VarDict = VarDict
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.worker_id = worker_id

        # Update checkpoint file with worker ID to ensure uniqueness
        if self.worker_id is not None:
            self.checkpoint_file = f"checkpoint_worker_{self.worker_id}.pkl"
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
        run = model.ParticleSystem(f"run_{sim_id}.particle", delPoints=1000)

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
            file_name = f"data_batch_worker_{self.worker_id}_{batch_id}.pkl"
        else:
            file_name = f"data_batch_{batch_id}.pkl"
        
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


def run_parallel_simulations(epoch, sampling_rate, VarDict, num_simulations, batch_size, num_workers):
    # Divide the total number of simulations among workers
    chunk_size = num_simulations // num_workers
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(run_batch_for_worker, epoch, sampling_rate, VarDict, start, end, batch_size, worker_id)
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

# Parameters for the simulation
epoch = 960.0
sampling_rate = 1
num_simulations = 100000  # Adjust the number of simulations if needed
batch_size = 50  # Batch size for saving
VarDict = {
    "PSD_slope_fast": "fast_a1", "PSD_break_fast": "fast_b1",
    "mu_fast": "fast_mu", "sig_fast": "fast_sig", "PSD_slope_slow": "a1", "PSD_break_slow": "b1",
    "mu_slow": "mu", "sig_slow": "sig", "B_0": "B_0", "gamma": "gamma", "ampfac": "ampfac",
    "size_0": "size_0", "noise_NIR": "vlt_noise", "X_offset": "I_offset", "rate_conv": "rate_conv",
    "eff_area": "I_eff_area", "model_gain": "a", "f0_B": "f_0_B", "f0_theta": "f_0_size",
    "noise_345GHz": "APEX_noise", "noise_230GHz": "SMA_noise", "noise_340GHz": "APEX_noise"
}

# Run the parallel simulations with 4 workers
num_workers = 4
run_parallel_simulations(epoch, sampling_rate, VarDict, num_simulations, batch_size, num_workers)
