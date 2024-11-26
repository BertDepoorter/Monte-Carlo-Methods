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
        self.boundary_condition = boundary_condition
        self.sampling_method = sampling_method

    def initialize_spins(self):
        '''
        Generate a random lattice with spins between 0, 1, ..., q-1

        input:
        - nothing, we just call this function. spin initialization is completely random

        output: 
        - N^2 array, with spins between 0 and q randomly assigned
        '''

        N2 = self.size**2
        q = self.q
        r = 0
        q_values = []
        while r < q:
            q_values.append(r)
            r += 1
        spins  = np.random.choice(q_values, size=N2)
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
