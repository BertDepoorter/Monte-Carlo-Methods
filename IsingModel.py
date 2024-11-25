import numpy as np
import matplolib.pyplot as plt
import pandas as pd
import scipy as sp

class IsingModel:
    """
    A class to represent the Ising model with customizable parameters.
    
    Attributes:
    - size (int): The size of the lattice (NxN).
    - J (float): Coupling constant.
    - h (float): External magnetic field.
    - boundary_condition (str): The type of boundary condition ("helical" or other future options).
    - sampling_method (str): The method of sampling ("uniform" for now).
    """
    
    def __init__(self, size, J=1, h=0, boundary_condition="helical", sampling_method="uniform"):
        """
        Initialize the Ising model.
        
        Parameters:
        - size (int): Size of the lattice (NxN).
        - J (float): Coupling constant (default is 1).
        - h (float): External magnetic field (default is 0).
        - boundary_condition (str): Type of boundary condition (default is "helical").
            options include 'helical', 'periodic'
        - sampling_method (str): Sampling method (default is "uniform").
            options include 'uniform', 'metropolis', 'hit and miss'.
        - seed (int, optional): Random seed for reproducibility.
        """
        self.size = size
        self.J = J
        self.h = h
        self.boundary_condition = boundary_condition
        self.sampling_method = sampling_method
    
    def initialize_spins(self):
        """
        Initialize a random spin configuration.
        
        Returns:
        - spins (np.ndarray): array of spins in {-1, 1}.
                shape depends on the choice of boundary conditions
        """
        spins = np.random.choice([-1, 1], size=self.size**2) 
        if self.boundary_condition == 'helical':
            return spins
        elif self.boundary_condition == 'periodic':
            return np.reshape(spins, (self.size, self.size))
    
    def get_neighbors(self, indices):
        """
        Get the neighbors of a spin at index i using helical boundary conditions.
        
        Parameters:
        - indices (list): indices of the spin. 
        
        Returns:
        - neighbors (list of int): Indices of neighboring spins.
        """

        if self.boundary_condition == 'helical':
            i = indices[0]
            N = self.size
            N2 = N**2
            neighbors = [
                (i + 1) % N + (i // N) * N,  # Right neighbor
                (i - 1 + N) % N + (i // N) * N,  # Left neighbor
                (i + N + N2) % N2,  # Below neighbor
                (i - N + N2) % N2,  # Above neighbor
            ]
            return neighbors

        elif self.boundary_condition == 'periodic':
            i = indices[0]
            j = indices[1]
            N = self.size
            N2 = N**2
            neighbours = [
                [i, (j+1+N)%N]       # right neighbour
                [i, (j-1+N)%N]       # left neighbour
                [(i-1+N)%N, j]       # below neighbour
                [(i+1+N)%N, j]       # Above neighbour
            ]
            return neighbours
    
    def calculate_energy(self, spins):
        """
        Calculate the energy of a spin configuration.
        
        Parameters:
        - spins (np.ndarray): 1D array of spins.
        
        Returns:
        - energy (float): Total energy of the configuration.
        """
        N = self.size
        energy = 0

        if self.boundary_condition == 'helical':
            for i in range(N**2):
                neighbors = self.get_neighbors([i])
                energy -= self.J * spins[i] * sum(spins[j] for j in neighbors)  # Interaction term
                energy -= self.h * spins[i]  # Magnetic field term
            return energy / 2  # Divide by 2 to correct for double counting

        elif self.boundary_condition == 'periodic':
            for i in range(N):
                for j in range(N):
                    neighbours = self.get_neighbours([i, j])
                    for pair in neighbours:
                        pair_i = pair[0]
                        pair_j = pair[1]
                        energy -= self.J*spins[i, j]*spins[pair_i, pair_j]      # energy from neighbours interaction
                        energy -= self.h*spins[i, j]        # energy by magnetic field 
                    return energy

    
    def sample_energies(self, num_samples):
        """
        Sample energies of random configurations.
        
        Parameters:
        - num_samples (int): Number of random configurations to sample.
        
        Returns:
        - energies (list of float): Energies of the sampled configurations.
        """

        if self.sampling_method != "uniform" or 'metropolis' or 'hit and miss':
            raise NotImplementedError(f"Sampling method '{self.sampling_method}' is not implemented.")

        # Implement uniform sampling method
        if self.sampling_method == 'uniform':
            energies = []
            for _ in range(num_samples):
                spins = self.initialize_spins()
                energy = self.calculate_energy(spins)
                energies.append(energy)
            return energies

        # Implement Metropolis algorithm 
        elif self.sampling_method == 'metropolis':
            energies = []
            spins = self.initialize_spins()
            for _ in range(num_samples):
                if self.boundary_condition != 'helical':
                    raise NotImplementedError
                i = np.random.randint(0, self.size**2, 1)
                spin = spins[i]
                E_old = self.calculate_energy(spins)
                spins[i] = spins[i]*(-1)
                E_new = self.calculate_energy(spins)
                delta_E = E_new - E_old
                if delta_E >0:
                    r = np.random.uniform(0,1, 1)
                    beta = 1/(self.kB*self.T)
                    Boltzmann_factor = np.exp(-beta*delta_E)
                    if r > Boltzmann_factor:
                        # flip the spin at place i again.
                        spins[i] = spins[i]*(-1)
                energies.append(energy)
            return energies

        elif self.sampling_method == 'hit and miss':
            energies = []
            raise NotImplementedError

    def energy_normalized(self, energies):
        '''
        Convert array of energies to energy per bond: e = E/2N^2

        Parameters: 
        - array of energies

        Returns:
        - array with energy per bond
        '''
        return energies/(2*self.size**2)

    def visualize_energy(self, energies):
        '''
        Functionality to plot the variation of the sampled energies

        Parameters: 
        - energies: array containing all the different sampled energies

        returns: 
        - visualization of the energy fluctuations under random sampling
        '''
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        samples = np.linspace(0, len(energies), len(energies))
        ax.set_xlabel('Samples', fontsize=14)
        ax.set_ylabel('Energy [ J ]', fontsize=14)
        title = 'Energy for samples, J = '+str(self.J)
        ax.plot(samples, energies, color='blue', alpha=0.7, label='samples')
        ax.legend()
        plt.axhline(0, color='gray', linestyle='dashed')
        ax.set_title(title, fontsize=17)
        plt.tight_layout()
        plt.show()

    def make_histogram(self, energies, bins=25, normalize=False):
        '''
        Functionality to plot histogram of the sampled energies

        Input:
        - energies (list): contains the energies of all sampled configurations
        - bins (int): how many bins for the histogram
        - normalize (bool): do we normalize the data or not. Default is False. 
                If True, we also plot gaussian distribution arond mean of samples 
        
        Output:
        - figure
        '''

        fig, ax = plt.subplots(1,1, figsize=(8,6))
        title= 'Histogram of sampled energy, J = '+str(self.J)
        ax.hist(energies, bins=bins)
        if normalize:
            return NotImplementedError
            ax.setylabel('Normalized counts', fontsize=14)
        ax.set_xlabel('Sampled energy [ J ]')
        ax.set_ylabel('Counts', fontsize=14)
        ax.set_title(title, fontsize=1)
