import numpy as np
import matplotlib.pyplot as plt
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
    
    def __init__(self, size, J=1, h=0, T=1, boundary_condition="helical", sampling_method="uniform"):
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
        self.T = T
        self.kB = 1 #e-23, we set this to 1 to have illustrate the behaviour nicely and don't have to be concerned around precision and such
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
                [i, int((j+1+N)%N)]       # right neighbour
                [i, int((j-1+N)%N)]       # left neighbour
                [int((i-1+N)%N), j]       # below neighbour
                [int((i+1+N)%N), j]       # Above neighbour
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

        # if self.sampling_method != "uniform" or 'metropolis' or 'hit and miss':
            # raise NotImplementedError(f"Sampling method '{self.sampling_method}' is not implemented.")

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
                        energies.append(E_old)
                    else: energies.append(E_new)
            return energies

        elif self.sampling_method == 'hit and miss':
            energies = []
            N = self.size
            N2 = self.size**2
            ground_state = -2*N2
            
            for _ in range(num_samples):
                spins = self.initialize_spins()
                beta = 1/(self.kB*self.T)
                E = self.calculate_energy(spins)
                r = np.random.uniform(0, 1, 1)
                Boltzmann_factor = np.exp(-beta*(E-ground_state))
                if r <= Boltzmann_factor:
                    energies.append(E)
            return energies

    def sample_spin_configurations(self, num_samples):
        """
        Sample spin configurations and return these as a 2D array.
        
        Parameters:
        - num_samples (int): Number of random configurations to sample.
        
        Returns:
        - spins (list of arrays): sampled spin configurations.
        """

        # if self.sampling_method != "uniform" or 'metropolis' or 'hit and miss':
            # raise NotImplementedError(f"Sampling method '{self.sampling_method}' is not implemented.")

        # Implement uniform sampling method
        if self.sampling_method == 'uniform':
            spins_sampled = np.asarray([])
            for _ in range(num_samples):
                spins = self.initialize_spins()
                spins_sampled = np.vstack([spins_sampled, spins])
            return spins_sampled

        # Implement Metropolis algorithm 
        elif self.sampling_method == 'metropolis':
            spins_sampled = np.zeros(100)
            spins = self.initialize_spins()
            for _ in range(num_samples):
                if self.boundary_condition != 'helical':
                    raise NotImplementedError
                i = np.random.randint(0, self.size**2, 1)
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
                spins_sampled = np.vstack([spins_sampled, spins])
            return spins_sampled

        elif self.sampling_method == 'hit and miss':
            spins_sampled = np.vstack([])
            N = self.size
            N2 = self.size**2
            ground_state = -2*N2
            
            for _ in range(num_samples):
                spins = self.initialize_spins()
                beta = 1/(self.kB*self.T)
                E = self.calculate_energy(spins)
                r = np.random.uniform(0, 1, 1)
                Boltzmann_factor = np.exp(-beta*(E-ground_state))
                if r <= Boltzmann_factor:
                    spins_sampled = np.vstack([spins_sampled, spins])
            return spins_sampled


    def energy_normalized(self, energies):
        '''
        Convert array of energies to energy per bond: e = E/2N^2

        Parameters: 
        - array of energies

        Returns:
        - array with energy per bond
        '''
        return np.asarray(energies)/(2*self.size**2)

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

    def make_histogram(self, energies, bins=25, normalize=False, magnetization=False):
        '''
        Functionality to plot histogram of the sampled energies / magnetizations

        Input:
        - energies_magnetizations (list): contains the energies or magnetizations of all sampled configurations
        - bins (int): how many bins for the histogram
        - normalize (bool): do we normalize the data or not. Default is False. 
                If True, we also plot gaussian distribution arond mean of samples 
        
        Output:
        - histogram of the sampled energies or magnetizations
        '''
        if magnetization == False:
            fig, ax = plt.subplots(1,1, figsize=(8,6))
            title= 'Histogram of sampled energy, J = '+str(self.J)
            if normalize == False:
                ax.hist(energies, bins=bins)
                ax.set_ylabel('Counts', fontsize=14)
            if normalize == True:
                # energies_norm = self.energy_normalized(energies)
                ax.hist(energies, bins=bins, density=True, label='normalized density')
                mu = np.sum(energies)/len(energies)
                sigma = np.std(energies)
                energy_min = np.min(energies)
                energy_max = np.max(energies)
                energy_array = np.linspace(energy_min, energy_max, 1000)
                gaussian = sp.stats.norm.pdf(energy_array, loc=mu, scale=sigma)
                ax.plot(energy_array, gaussian, color='red', label='Normal distribution')
                ax.legend()
                ax.set_ylabel('Normalized counts', fontsize=14)
            ax.set_xlabel('Sampled energy [ J ]')
            ax.set_title(title, fontsize=1)

        elif magnetization == True:
            fig, ax = plt.subplots(1,1, figsize=(8,6))
            title= 'Histogram of sampled magnetizations, J = '+str(self.J)
            if normalize == False:
                ax.hist(energies, bins=bins)
                ax.set_ylabel('Counts', fontsize=14)
            if normalize == True:
                ax.hist(energies, bins=bins, density=True)
                mu = np.sum(energies)/len(energies)
                sigma = np.std(energies)
                energy_min = np.min(energies)
                energy_max = np.max(energies)
                energy_array = np.linspace(energy_min, energy_max, 1000)
                gaussian = sp.stats.norm.pdf(energy_array, loc=mu, scale=sigma)
                ax.plot(energy_array, gaussian, color='red', label='Normal distribution')

                ax.set_ylabel('Normalized counts', fontsize=14)
            ax.set_xlabel('Sampled magnetization [ J ]')
            ax.set_title(title, fontsize=1)


    def get_magnetization(self, spins):
        '''
        Function to calculate the total magnetization of a certain configuration of spins

        input:
        - spins (array): array containing all the generated spin configurations

        output:
        - Total magnetization of each of the spin configurations
        '''
        M = np.zeros(len(spins))
        N2= self.size**2
        for k in range(len(spins)):
            print('shape of element in spin_samples: ', spins[k].shape, type(spins[k]))
            M_sample = 1/N2*np.sum(spins[k])
            M.append(M_sample)
        return M

    def get_exact_magnetization(self):
        '''
        Function that returns the exact magnetization for the Ising model with a certain temperature and J value
        This has been computed analytically.

        input:
        - self: only J, T are needed

        output:
        - exact magnetization value
        '''

        M = (1-np.sinh(2*self.J/self.T)**(-4))**(1/8)
        return M

    def plot_magnetization(self, num_sweeps, chains=1, spins=[]):
        '''
        Function that creates plot like fig 3 in the lecture notes

        input:
        - num_sweeps (int): number of spin MC steps to take (in sweeps!!) 
        - chains (int): how many Monte Carlo chains we run. Each chain has the same number of sweeps
        - spins (list, optional): if we have list of generated spin configurations, we can skip the generation of the spin lattices

        output: 
        - plot of the magnetization, with the exact magnetization plotted as a baseline
        '''

        if self.sampling_method != 'metropolis':
            raise NotImplementedError

        N = self.size
        N2 = self.size**2
        num_samples = num_sweeps*N2
        M_exact = self.get_exact_magnetization()
        magnetizations = np.zeros(num_samples)
        sweeps = np.linspace(0, num_samples, num_samples)/N2
        
        # Create figure instances
        fig, ax = plt.subplots(1,1, figsize=(10, 7))
        
        if spins == []:
            # run several chains
            for i in range(chains):
                # generate num_samples spin configurations with desired algorithm
                samples = self.sample_spin_configurations(num_samples)[1:]
                magnetizations = np.add(magnetizations, self.get_magnetization(samples))
                lab = 'Sampled magnetizations, chain '+str(i+1)
                ax.plot(sweeps, magnetizations, alpha=0.7, label=lab)
                magnetizations = np.zeros(num_samples)
        else:
            # calculate magnetizatio for each of the given spin configurations
            magnetizations = np.add(magnetizations, self.get_magnetization(spins))
            ax.plot(sweeps, magnetizations, color='red', alpha=0.7, label='Sampled magnetizations')

        ax.axhline(M_exact, color='black', alpha=0.7, linestyle='dashed', label='Exact Magnetization')
        print(M_exact)
        ax.set_xlabel('Number of MC sweeps', fontsize=14)
        ax.set_ylabel('Magnetization', fontsize=14)
        fig.suptitle('Metropolis algorithm', fontsize=20)
        subtitle = 'T = '+str(self.T)+' , N = ' + str(self.size)
        ax.set_title(subtitle, fontsize=17)
        ax.legend()
        plt.show()
    
    def autocorrelation(self, t, tau_eq, magnetizations):
        '''
        calculates the autocorrelation function for some time t>tau_eq. 

        input:
        - t (int): index of MC step at which we want to know the autocorrelation function
        - tau_eq (int): index of MC step at which system has equilibrated. 
            Should be estimated from the plot of the magnetizations
        - magnetizations (list): list with all magnetization values

        output:
        - autocorrelation function as defined in eq 100 of lecture notes
        '''
        average_m = np.sum(magnetizations)/len(magnetizations)
        tf = len(magnetizations)
        chi = 0
        for s in range(tau_eq, tf):
            chi += (magnetizations[s]-average_m)*(magnetizations[s+t]-average_m)
        chi = chi/tf
        return chi