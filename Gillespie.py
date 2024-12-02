import numpy as np
import matplotlib.pyplot as plt

class EvolutionGillespie:
    """
    Class that simulates the evolution of a certain system, described by a set of 'chemical' reactions.

    Attributes:
    - number of different species present in the model (N)
    - list of reaction rates (coeff)
    - 3D array of reaction coefficients (reactions)
    - name (title): title of the system used for plots
    """

    def __init__(self, number_species, coeff, reactions, title):
        """
        Initialize the system.

        Parameters:
        - number_species (int): Number of species in the system.
        - coeff (list): List of reaction rates.
        - reactions (3D array): Reaction coefficients:
            * 1st dimension: 2 -> ingoing and outgoing coefficients.
            * 2nd dimension: Number of reaction equations present.
            * 3rd dimension: Number of species present (stoichiometric coefficients).
        - title (str): Title for plots.
        """
        self.N = number_species
        self.coeff = np.array(coeff)
        self.reactions = np.array(reactions)
        self.title = title

    def simulate(self, initial_state, t_max):
        """
        Simulate the system using Gillespie's algorithm.

        Parameters:
        - initial_state (list): Initial population of each species.
        - t_max (float): Maximum simulation time.

        Returns:
        - times (list): Time points.
        - populations (list of lists): Populations of each species at each time point.
        """
        # Initial setup
        state = np.array(initial_state)
        times = [0]
        populations = [state.tolist()]

        while times[-1] < t_max:
            # Calculate propensities for each reaction
            propensities = []
            for r in range(len(self.coeff)):
                rate = self.coeff[r]
                ingoing = self.reactions[0, r]
                propensity = rate
                for s in range(self.N):
                    if ingoing[s] > 0:
                        propensity *= np.math.comb(state[s], ingoing[s])  # n choose k
                propensities.append(propensity)

            total_propensity = sum(propensities)
            if total_propensity == 0:
                break  # No reactions can occur

            # Time until next reaction
            tau = np.random.exponential(1 / total_propensity)
            
            # Choose which reaction occurs
            reaction_index = np.random.choice(
                len(propensities), p=np.array(propensities) / total_propensity
            )

            # Update the state
            state += self.reactions[1, reaction_index] - self.reactions[0, reaction_index]
            times.append(times[-1] + tau)
            populations.append(state.tolist())

        return times, populations

    def plot_species(self, times, populations, species1, species2):
        """
        Plot the evolution of two species over time.

        Parameters:
        - times (list): Time points from the simulation.
        - populations (list of lists): Populations of each species at each time point.
        - species1 (int): Index of the first species to plot.
        - species2 (int): Index of the second species to plot.
        """
        populations = np.array(populations)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, populations[:, species1], label=f"Species {species1+1}")
        ax.plot(times, populations[:, species2], label=f"Species {species2+1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.set_title(self.title)
        ax.legend()
        ax.grid()
        fig.savefig('PLots/ex_Gillespie_'+self.title+'.png', dpi=300)
        return fig, ax

    def plot_species_against_each_other(self, populations, species1, species2):
        '''
        Function that plots two populations against each other. 
        We have no time axis, we plot species 1 on the x-axis and species 2 on the y-axis

        input: 
        - populations (list of lists): Populations of each species at each time point.
        - species1 (int): Index of the first species to plot.
        - species2 (int): Index of the second species to plot.
        '''
        fig, ax = plt.subplots(1,1, figsize=(7,7))
        populations = np.array(populations)
        ax.plot(populations[:, species1], populations[:, species2], color='red', label='Trajectory of populations')
        ax.scatter(populations[0, species1], populations[0, species2], s=50, color='black', label='Starting point')
        ax.scatter(populations[-1, species1], populations[-1, species2], s=50, color='blue', label='End point')
        ax.set_xlabel('Species 1', fontsize=14)
        ax.set_ylabel('Species 2', fontsize=14)
        ax.set_title(self.title, fontsize=20)
        ax.legend()
        fig.savefig('PLots/ex_Gillespie_noTime_'+self.title+'.png', dpi=300)
        return fig, ax