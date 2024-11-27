import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

class KawasakiModel:
    '''
    General class that allows simulations of the conserved order parameter Ising model

    Attributes include:
    - size_x (int): size of the lattice in x-direction
    - size_y (int): size of the lattice in y-direction
    - J (float): Coupling constant. 
        Default is 1
    - h (float): External magnetic field. 
        Default is 0
    - boundary_condition (str): The type of boundary condition ("helical" or other future options). 
        Default = 'helical'
    - local (bool): specify whether we want to use the local or non-local Kawasaki algorithm
        Default is True
    '''

    def __init__(self, size_x, size_y, J=1, h=0, T=1, boundary_condition='periodic', local=True):
        '''
        Initialize the class with the necessary attributes

        Parameters:
        - size (int): Size of the lattice (NxN).
        - q (int): number of spin levels
        - J (float): Coupling constant (default is 1).
        - h (float): External magnetic field (default is 0).
        - boundary_condition (str): Type of boundary condition (default is "helical").
            options include 'helical', 'periodic'
        - local (bool): Local Kawasaki algorithm or non-local version
            default is True, local version
        '''

        self.Lx = size_x
        self.Ly = size_y
        self.J = J
        self.h = h
        self.T = T
        self.kB = 1 # set to 1 to get normal numbers
        self.beta = 1/(self.T*self.kB)
        self.boundary_condition = boundary_condition
        self.local = local

    
