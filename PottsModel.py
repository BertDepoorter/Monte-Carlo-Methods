import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

class PottsModel:
    """
    A class to represent the q-dimensional Potts model with customizable parameters.
    
    Attributes:
    - size (int): The size of the lattice (NxN).
    - q (int): number of spin levels in Potts model
    - J (float): Coupling constant. 
        Default is 1
    - h (float): External magnetic field. 
        Default is 0
    - boundary_condition (str): The type of boundary condition ("helical" or other future options). 
        Default = 'helical'
    - sampling_method (str): The method of sampling ("uniform" for now). 
        default is 'metropolis'
    """

    def __init__(self, size, q,  J=1, h=0, T=1, boundary_condition="helical", sampling_method="uniform"):
        """
        Initialize the Potts model.
        
        Parameters:
        - size (int): Size of the lattice (NxN).
        - q (int): number of spin levels
        - J (float): Coupling constant (default is 1).
        - h (float): External magnetic field (default is 0).
        - boundary_condition (str): Type of boundary condition (default is "helical").
            options include 'helical', 'periodic'
        - sampling_method (str): Sampling method (default is "uniform").
            options include 'uniform', 'metropolis', 'hit and miss'.
        - seed (int, optional): Random seed for reproducibility.
        """
        self.size = size
        self.q = q
        self.J = J
        self.h = h
        self.T = T
        self.kB = 1 #e-23, we set this to 1 to have illustrate the behaviour nicely and don't have to be concerned around precision and such
        self.beta = 1/(self.kB*self.T)
        self.boundary_condition = boundary_condition
        self.sampling_method = sampling_method
        self.Boltzmann = self._precompute_boltzmann_factors()

        # create list with possible spin values
        r = 0
        q_values = []
        while r < self.q:
            q_values.append(r)
            r += 1
        self.q_values = q_values

    def initialize_spins(self):
        '''
        Generate a random lattice with spins between 0, 1, ..., q-1

        input:
        - nothing, we just call this function. spin initialization is completely random

        output: 
        - N^2 array, with spins between 0 and q randomly assigned
        '''

        N2 = self.size**2
        spins  = np.random.choice(self.q_values, size=N2)
        return spins
    
    def calculate_energy(self, spins):
        '''
        Calculate the energy of a Potts model for a given spin configuration

        input:
        - spins (array): configuration of spins

        output:
        - energy (float): energy of the given spin configuration
        '''
        E = 0
        for i in range(self.size**2):
            neighbours = self.get_neighbours(i)
            for neighbour in neighbours:
                if spins[neighbour] == spins[i]:
                    E -= 1
        return E / 2


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
                [i, int((j+1+N)%N)]       # right neighbour
                [i, int((j-1+N)%N)]       # left neighbour
                [int((i-1+N)%N), j]       # below neighbour
                [int((i+1+N)%N), j]       # Above neighbour
            ]
            return neighbours

    def sample_spin_configurations(self, num_samples):
        '''
        sample a specified number of spin samples 

        input:
        - num_samples (int): the number of samples

        output:
        - array of all sampled spin configurations
        '''

        samples = []
        spins = self.initialize_spins()
        samples = np.asarray(samples.append(spins))
        if self.sampling_method == 'heat bath':
            Boltzmann_factors = self._precompute_boltzmann_factors()
            for _ in range(num_samples):
                spins = self._sample_heat_bath(spins, Boltzmann_factors)
                samples = np.vstack([samples, spins])
        elif self.sampling_method == 'metropolis':
            for _ in range(num_samples):
                spins = self._update_metropolis(spins)
                samples = np.vstack([samples, spins])
        else:
            raise ValueError("Invalid method. Choose either 'heat_bath' or 'metropolis'.")
        return samples
    
    def _precompute_boltzmann_factors(self):
        """
        Precompute the Boltzmann factors for all possible neighbor energy contributions.

        output:
        dict: A dictionary mapping energy contributions to their Boltzmann factors.
        """
        boltzmann_factors = {}
        for energy in range(0, 5):  # Max energy contribution is 4 (all neighbors agree)
            boltzmann_factors[energy] = np.exp(self.beta * energy*self.J)
        return boltzmann_factors

    
    def _sample_heat_bath(self, spins):
        '''
        Implement the heat bath algorithm for the Potts Model

        input:
        - spins (array): array containing the spins

        output:
        - spins (array): the updated spin configuration with one heat bath step
        '''
        # choose spin to flip
        spin = np.random.randint(0, self.size*2, 1)
        
        # Calculate the weights for the different spin flips
        weights = self._get_weights(spins, spin) 

        #select spin flip with weighted probability
        new_spin_value =np.random.choice(self.q_values, 1, p=weights)

        # Update the spin value to the selected one
        # spin flip is always accepted
        spins[spin] = new_spin_value
        return spins
        

    def _get_weights(self, spins, spin):
        '''
        Function that calculates the weights assigned to different spin picks
        Now the acceptance probablility is always one, but the selection probability is non-uniform for different states
        '''
        neighbours = self.get_neighbors(spin)
        # get energy differences for 
        E_old = self.calculate_energy(spins)

        weights = []
        for q in range(self.q):
            neighbour_spins = [spins[neighbour] for neighbour in neighbours]
            delta_E = 0
            # get energy for nearest neighbours configuration
            for i in range(4):
                if float(neighbour_spins) == float(q):
                    delta_E += -self.J
            # get corresponding Boltzmann factor and add to weights

            weights.append(float(delta_E))
        weights = np.asarray(weights)/(self.q*self.size**2*sum(self.Boltzmann.values()))
        return weights






    def _update_heat_bath(self):
        """
        Perform a single heat bath update on the lattice using precomputed Boltzmann factors.
        """
        L = self.L
        for _ in range(L * L):  # Attempt to update each spin once on average
            x, y = np.random.randint(0, L), np.random.randint(0, L)  # Random spin
            neighbor_states = self._neighbor_states(x, y)
            neighbor_count = np.zeros(self.q, dtype=int)  # Count occurrences of each spin state
            
            # Count neighbor contributions
            for state in neighbor_states:
                neighbor_count[state] += 1

            # Calculate weights using precomputed Boltzmann factors
            weights = [
                self.boltzmann_factors[neighbor_count[s]] for s in range(self.q)
            ]
            weights = np.array(weights) / np.sum(weights)  # Normalize to form probabilities

            # Sample a new spin state based on the probabilities
            self.lattice[x, y] = np.random.choice(np.arange(self.q), p=weights)

        