import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, size, q,  J=1, h=0, T=1, boundary_condition="helical", sampling_method="uniform", init_zero=False):
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
        self.init_zero = init_zero

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
        if self.init_zero == False:
            N2 = self.size**2
            spins  = np.random.choice(self.q_values, size=N2)
            return spins
        elif self.init_zero == True:
            N2 = self.size**2
            spins  = np.zeros(N2)
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
            neighbours = self.get_neighbors(i)
            for neighbour in neighbours:
                if spins[neighbour] == spins[i]:
                    E -= 1
        return E / 2


    def get_neighbors(self, i):
        """
        Get the neighbors of a spin at index i using helical boundary conditions.
        
        Parameters:
        - indices (list): indices of the spin. 
        
        Returns:
        - neighbors (list of int): Indices of neighboring spins.
        """

        if self.boundary_condition == 'helical':
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

        spins = self.initialize_spins()
        samples = np.asarray(spins)
        if self.sampling_method == 'heat bath':
            for _ in range(num_samples):
                spins = self._sample_heat_bath(spins)
                samples = np.vstack([samples, spins])
        elif self.sampling_method == 'metropolis':
            for _ in range(num_samples):
                spins = self._sample_metropolis(spins)
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
        Implement the heat bath algorithm for the Potts Model. Calling this function corresponds to a single spin flip

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

        input: 
        - spins (array): current spin configuration
        - spin (int): selected spin that we want to flip

        output:
        - weights (array): array containing the non-uniform weights assigned to the spin flip
        '''
        neighbours = self.get_neighbors(spin)
        neighbour_spins = [spins[neighbour] for neighbour in neighbours]

        weights = []
        for q in range(self.q):
            # Count energy connections
            energy_connections = 0
            for i in range(4): # loop over neighbour spins
                if float(neighbour_spins[i]) == float(q):
                    energy_connections += 1
            # get corresponding Boltzmann factor and add to weights

            weights.append(self.Boltzmann[energy_connections])
        weights = np.asarray(weights)
        weights = weights/(np.sum(weights))
        # In notes this states that we need to divide with q*M, where M is total number of spins in the lattice
        # However this does not result in weights that sum to 1.
    
        return weights

    def _sample_metropolis(self, spins):
        '''
        Implement the heat bath algorithm for the Potts Model. Calling this function corresponds to a single spin flip

        input:
        - spins (array): array containing the spins

        output:
        - spins (array): the updated spin configuration with one heat bath step
        '''
        # choose spin to flip
        spin = np.random.randint(0, self.size*2, 1)
    
        # Select randomly a new spin, exclude own value
        spin_proposed =spins[spin]
        while spin_proposed == spins[spin]:
            spin_proposed = np.random.choice(self.q_values, 1)

        # Run Monte_carlo;
        #calculate energy difference
        neighbours = self.get_neighbors(spin)
        neighbour_spins = [spins[neighbour] for neighbour in neighbours]
        connections_old = 0
        connections_new = 0
        for i in range(4):
            if neighbour_spins[i] == spins[spin]:
                connections_old += 1
            if neighbour_spins[i] == spin_proposed:
                connections_new += 1
        delta_E = (connections_new-connections_old)*(-self.J)
        if delta_E <= 0:
            spins[spin] = spin_proposed
        else:
            r = np.random.uniform(0,1, 1)
            if r <= self.Boltzmann[int(delta_E/(self.J))]:
                # accept the spin flip
                spins[spin] = spin_proposed
            else:
                # Reject spin flip, so do nothing
                pass
        return spins
    
    def get_magnetization(self, spins):
        '''
        Function that calculates the magnetization for a given spin configuration
        
        input:
        - spins: array containing the lattice full of spins

        output:
         - magnetization (float): total magnetization of the lattice
        '''
        N2 = self.size**2
        M = 1/N2*np.sum(spins)
        return M
    
    def magnetization_chain(self, chain):
        '''
        Calculate the magnetization of certain number of spin samples

        input: 
        - chain: array containing the different spin configurations

        output:
        - magnetization array with all magnetizations for the different spin configurations
        '''
        num_samples = len(chain)
        magnetizations = np.zeros(num_samples)
        for i in range(num_samples):
            magnetizations[i] += self.get_magnetization(chain[i])
        return magnetizations


    def plot_magnetizations(self, num_samples, chains=1, spin_samples=[], chain_labels=None):
        '''
        Function to make a plot of the magnetization of the spin lattice

        input:
        - num_samples(int): number of spin configurations to sample
        - chains (int, optional): number of MC chains to run. 
            Default is one.
        - spin_samples (array of array of arrays, optional): optional argument to apply plotting functionality on an already sampled set of spin configurations
            dimension should be 3: 1st dimension are the different chains, 2nd dimension is the array containing the different spin configurations (also arrays) in each chain.
        - chain_labels: list of chain labels. These contain the labels to assign to the different chains we put in

        output:
        - Plot of the magetizations
        '''

        # Sample chains if not specified
        fig, ax = plt.subplots(1,1, figsize=(10,7))

        if spin_samples == []:
            sweeps = np.linspace(0, num_samples, num_samples)/self.size**2 #array for number of sweeps
            for chain in range(chains):
                new_chain = self.sample_spin_configurations(num_samples)
                magnetizations = self.magnetization_chain(new_chain)
                label = 'Chain '+str(chain)
                ax.plot(sweeps, magnetizations, label=label)
        else:
            if spin_samples.ndim != 3:
                return ValueError
            else: 
                for i in range(len(spin_samples)):
                    # get magnetizations for each chain
                    length = len(spin_samples[i])
                    sweeps = np.linspace(0, length, length)/self.size**2 #array for number of sweeps
                    magnetizations = np.zeros(len(spin_samples[i]))
                    for k in range(len(spin_samples[i])):
                        magnetizations[k] += self.get_magnetization(spin_samples[i][k])
                    # Plot magnetization of this chain
                    if chain_labels == None:
                        return ValueError
                    ax.plot(sweeps, magnetizations, label=chain_labels[i])
        
        ax.set_xlabel('MC sweeps (MC steps per lattice site)', fontsize=14)
        ax.set_ylabel('Magnetization', fontsize=14)
        fig.suptitle('Magnetization for '+str(self.q)+'-dimensional Potts Model', fontsize=20)
        ax.set_title('N = '+str(self.size)+', J = '+str(self.J)+', sample size = '+str(num_samples))
        ax.legend()
        fig.savefig('Plots/Magnetizations_Potts_model.png', dpi=300)
        