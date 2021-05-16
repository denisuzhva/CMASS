# simulator.py


from abc import ABCMeta, abstractmethod
from agent import *
import numpy as np


class Simulator:

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

    def __init__(self, n_agents, dt, adj_matrix, cluster_map, clust_time, rho, mus, f0s, nat_freqs, init_states=None):
        """
        """
        assert n_agents * n_agents == adj_matrix.size
        self.__time = 0.
        self.__dt = dt
        self.__clust_time = clust_time 
        self.__n_agents = n_agents
        self.__adj_matrix = adj_matrix
        self.__adj_dict = {i: np.nonzero(self.__adj_matrix[i])[0].tolist() for i in range(n_agents)}

        self.__agents = {i: OscillatorAgent(self, rho, nat_freqs[i], mus[i], f0s[cluster_map[i]], init_states[i] if isinstance(init_states, list) else None) for i in range(n_agents)}
        self.__agents_inv = {agent: i for i, agent in self.__agents.items()}

    def get_time(self):
        return self.__time

    def get_dt(self):
        return self.__dt

    def get_neighbor_outputs(self, agent):
        master_id = self.__agents_inv[agent]
        master_neighbors = self.__adj_dict[master_id]
        neighbor_outputs = [self.__agents[neighbor].send_output() for neighbor in master_neighbors]
        return neighbor_outputs

    def get_all_outputs(self):
        all_outputs = [agent.send_output() for agent in self.__agents.values()]
        return all_outputs

    def update_all_states(self):
        for i in range(self.__n_agents):
            do_cluster_control = True if self.__time >= self.__clust_time else False
            self.__agents[i].update_state(do_cluster_control)
        self.__time += self.__dt

