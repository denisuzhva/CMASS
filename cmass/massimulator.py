# simulator.py


from abc import ABCMeta, abstractmethod
from .agent import *
import numpy as np


class Simulator:
    """An interface for multiagent system simulator. All the methods below must be realized in a child class."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_time(self):
        pass 

    @abstractmethod
    def get_dt(self):
        pass

    @abstractmethod
    def get_neighbor_outputs(self, agent):
        pass

    @abstractmethod
    def get_all_outputs(self):
        pass

    @abstractmethod
    def update_all_states(self):
        pass


class KuramotoSimulator(Simulator):
    """A multiagent system simulator for Kuramoto model."""

    def __init__(self, n_agents, dt, adj_matrix, cluster_map, clust_time, rho, mus, f0s, nat_freqs, init_states=None):
        """
        Initialize a simulator.

        Args:
            n_agents: Number of agents
            dt: Time step for iterations
            adj_matrix: Adjacency matrix describing agent connectivity
            cluster_map: A map {agent : cluster}
            rho: Universal local control strength
            mus: Cluster control sensitivities for each agent
            f0s: Frequencies for each sinusoidal control input
            nat_freqs: Oscillator natural frequencies
            init_states: Set specific initial agent states 
        """
        assert n_agents * n_agents == adj_matrix.size # Check if match
        self.__time = 0.
        self.__dt = dt
        self.__clust_time = clust_time 
        self.__n_agents = n_agents
        self.__adj_matrix = adj_matrix
        self.__adj_dict = {i: np.nonzero(self.__adj_matrix[i])[0].tolist() for i in range(n_agents)}
        self.__agents = {i: OscillatorAgent(self, rho, mus[i], nat_freqs[i], f0s[cluster_map[i]], init_states[i] if isinstance(init_states, list) else None) for i in range(n_agents)}
        self.__agents_inv = {agent: i for i, agent in self.__agents.items()}

    def get_time(self):
        """
        Get current simulation time.

        Returns:
            self.__time: Current simulation time
        """
        return self.__time

    def get_dt(self):
        """
        Get simulation time step.

        Returns:
            self.__dt: Time step
        """
        return self.__dt

    def get_neighbor_outputs(self, agent):
        """
        Get outputs of neightbors of a specific agent

        Args:
            agent: An agent neighbors of which are asked for an output

        Returns:
            neighbor_outputs: Outputs of neighbors of a specified agent
        """
        master_id = self.__agents_inv[agent]
        master_neighbors = self.__adj_dict[master_id]
        neighbor_outputs = [self.__agents[neighbor].send_output() for neighbor in master_neighbors]
        return neighbor_outputs

    def get_all_outputs(self):
        """
        Get a list of outputs of all agents in the system.

        Returns:
            all_outputs: Outputs of all the agents
        """
        all_outputs = [agent.send_output() for agent in self.__agents.values()]
        return all_outputs

    def update_all_states(self):
        """Update states of all agents by calling update_state() and regulating cluster control."""
        for i in range(self.__n_agents):
            do_cluster_control = True if self.__time >= self.__clust_time else False
            self.__agents[i].update_state(do_cluster_control)
        self.__time += self.__dt # Increment iteration by the time step

