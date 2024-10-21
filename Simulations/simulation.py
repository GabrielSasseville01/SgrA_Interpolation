import pickle
import os
import lc_model as model
import matplotlib.pyplot as plt
import numpy as np

class Simulation:
    def __init__(self, epoch, sampling_rate, VarDict, num_simulations, batch_size=100, checkpoint_file="checkpoint.pkl"):
        self.epoch = epoch
        self.sampling_rate = sampling_rate
        self.VarDict = VarDict
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.start_simulation = 0
        self.batch_id = 0
        
        # Load checkpoint if exists
        self.load_checkpoint()

    def run_simulation(self):
        time = self.epoch * self.sampling_rate

        run = model.ParticleSystem(f"run.particle", delPoints=1000)

        run.ParamSet(self.VarDict, PickParticle=True)
        run.params_model.update({"noise_22GHz": 0.02})

        generator = model.DataGenerator(run.params_model, time)
        generator.ModelSetup()

        generator.LightCurveData(self.sampling_rate)
        generator.CalculateLightCurves(0, time)

        data = generator.masks(0.2)
        
        return data

    def save_data_batch(self, batch_data, batch_id):
        with open(f"data_batch_{batch_id}.pkl", "wb") as f:
            pickle.dump(batch_data, f)
        print(f"Saved data_batch_{batch_id}.pkl")

    def save_checkpoint(self):
        checkpoint_data = {
            "start_simulation": self.start_simulation,
            "batch_id": self.batch_id
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print("Checkpoint saved.")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
                self.start_simulation = checkpoint_data["start_simulation"]
                self.batch_id = checkpoint_data["batch_id"]
            print("Checkpoint loaded.")
        else:
            print("No checkpoint found. Starting from the beginning.")

    def run_and_save_simulations(self):
        batch_data = []

        for i in range(self.start_simulation, self.num_simulations):
            data = self.run_simulation()
            batch_data.append(data)

            # Save data in batches
            if (i + 1) % self.batch_size == 0:
                self.save_data_batch(batch_data, self.batch_id)
                batch_data = []
                self.batch_id += 1
                self.start_simulation = i + 1
                self.save_checkpoint()
        
        # Save any remaining data
        if batch_data:
            self.save_data_batch(batch_data, self.batch_id)
            self.start_simulation = self.num_simulations
            self.save_checkpoint()

    @staticmethod
    def load_data_batch(batch_id):
        with open(f"data_batch_{batch_id}.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_all_data(num_batches):
        all_data = []
        for batch_id in range(num_batches):
            batch_data = Simulation.load_data_batch(batch_id)
            all_data.extend(batch_data)
        return all_data
    
    @staticmethod
    def plot_simulation(dataset, sim_number):
        fig, axs = plt.subplots(4, 1, figsize=(12, 18))

        wavelengths = dataset.keys()
        for i, wavelength in enumerate(wavelengths):
            xdata_unmasked = dataset[wavelength]['xdata_unmasked']
            xdata_masked = dataset[wavelength]['xdata_masked']
            ydata_unmasked = dataset[wavelength]['ydata_unmasked']
            ydata_masked = dataset[wavelength]['ydata_masked']

            axs[i].scatter(xdata_unmasked, ydata_unmasked, label='Unmasked', s=5)
            axs[i].scatter(xdata_masked, ydata_masked, label='Masked', s=5)
            
            axs[i].set_title(f'{wavelength} Data')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Flux')
            axs[i].legend()

        fig.suptitle(f'Simulation {sim_number}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_all_simulations(all_data):
        for i, dataset in enumerate(all_data):
            Simulation.plot_simulation(dataset, i)