import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

class EvolutionGillespie:
    '''
    Class that simulates the evolution of a certain system, described by a set of 'chemical' reactions

    Attributes
    - number of different species present in model
    - list of reaction rates
    - 3D array of reaction coefficients
    - name (str) of the system: will be used to title plots
    '''

    def __init__(self, number_species, coeff, title):
        '''
        Initialize the system

        - number of species in system
        - list of reaction rates
        - 3D array for all reaction coefficients
            1st dimension: 2 -> ingoing and outgoing coefficients
            2nd dimension: number of reaction equations present
            3rd dimension: number of species present. contains stoichiometric coefficients
        - name (str): to title plots
        '''

        self.N = number_species
        self.coeff = coeff
        self.title = title

    def sample_evolution(self, num_samples, initial=[]):
        '''
        Function to simulate the evolution of the system under the specified dynamics

        input:
        - num_samples: how many steps to we simulate for

        output:
        - 2D array: containing all the species populations at all steps. 
            Dimension of array = num_samples x number_species
        '''
        if initial == []:
            init = self.initialize()


    def initialize(self):
        '''
        Initialize population levels randomly
        '''
        pass
