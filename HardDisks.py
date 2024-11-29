# Class to solve hard disks problem with MC

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

class HardDisks:
    '''
    General class to solve exercises concerning the hard disks problem 4.12 in the lecture notes

    contains functionality to sample accroding to metropolis algorithm as well as plotting functionality
    '''

    def __init__(self, N, q, L, sigma,  algorithm='metropolis', initial_conf='rectangular'):
        '''
        Initialize the grid with the hard spheres.

        input:
        - N: numbers of particles in system
        - q (float): characteristic length for random step
        - L (int): size of the lattice
        - sigma (float): characterizes hard spheres potential
        - algorithm (str): algorithm with which we update the configuration
        - initial_conf (str): how to choose initial configuration 
            options are 'rectangular' and 'random'
        '''
        self.N = N
        self.q = q
        self.L = L
        self.algorithm = algorithm
        self.initial_conf = initial_conf

        # Generate initial configuration
        grid_initial = self.initialize()
        return grid_initial

    def initialize(self):
        '''
        Initialize the grid with specified method
        ''' 
        if self.initial_conf == 'rectangular':
            if float(np.floor(np.sqrt(self.N))) != 0.0:
                return ValueError
            one_line = np.sqrt(self.N)
            dist = self.L/(one_line-1)
            grid = np.zeros((one_line, one_line, 2))
            for i in range(one_line):      #loop over rows
                for j in range(one_line):  # loop over columnns
                    grid[i,j] += np.asarray([i, j]*dist)
            return grid
        elif self.initial_conf == 'random':
            grid = np.zeros(,2)
            for i in range(self.N):    #assign each particle a random position
                x = np.random.uniform(0, L, 1)
                y = np.random.uniform(0, L, 1)
                position = np.asarray([x, y])
            
    def get_distance(self, position_a, position_b):
        '''
        Calculate the distance between two particles considering periodic boundary conditions.

        input: 
        - position_a (array): Position of particle A (shape: (2,))
        - position_b (array): Position of particle B (shape: (2,))

        output:
        - distance (float): Euclidean distance between the particles.
        '''
        delta = np.abs(position_a - position_b)
        delta = np.where(delta > self.L / 2, self.L - delta, delta)  # Apply PBC
        return np.sqrt((delta**2).sum())

    def is_valid_move(self, new_position, particle_index):
        '''
        Check if a proposed move is valid (i.e., does not cause overlap).

        input:
        - new_position (array): Proposed new position of the particle (shape: (2,))
        - particle_index (int): Index of the particle being moved

        output:
        - valid (bool): True if move is valid, False otherwise
        '''
        for i, pos in enumerate(self.positions):
            if i != particle_index:  # Exclude self-check
                if self.get_distance(new_position, pos) < 2 * self.sigma:
                    return False
        return True
    
    def simulate(self, num_samples):
        '''
        Function to sample new configurations using the metropolis algorithm

        input:
        - num_samples (int): number of times to sample a displacement

        output:
        - final configuration (grid of samples)
        - 
        '''

        for step in range(num_samples):
            self.metropolis_step()
        
    def metropolis_step(self):
        '''
        Perform a single Metropolis step.

        - Randomly selects a particle
        - Proposes a random move
        - Accepts or rejects the move based on overlap constraints
        '''
        # Step 1: Pick a random particle
        particle_index = np.random.randint(self.N)
        old_position = self.positions[particle_index]

        # Step 2: Propose a move
        delta = (np.random.rand(2) - 0.5) * 2 * self.q  # Random shift in [-q, q] range
        new_position = old_position + delta

        # Apply periodic boundary conditions
        new_position = new_position % self.L

        # Step 3: Check validity of the move
        if self.is_valid_move(new_position, particle_index):
            # Accept the move
            self.positions[particle_index] = new_position
        
    def acceptance_rate(self, num_samples):
        ''''
        Function that returns the average acceptance rate for the metropolis sampling
        '''
