# agent.py


from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.core.numeric import full
from .controller import *


class Agent(metaclass=ABCMeta):
    """An interface for agents. All the methods below must be realized in a child class."""

    @abstractmethod
    def __init__(self, parameters):
        pass

    @abstractmethod
    def send_output(self):
        pass

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def change_local_controller(self, new_local_controller):
        pass

    @abstractmethod
    def change_cluster_controller(self, new_cluster_controller):
        pass


class OscillatorAgent(Agent):
    """An oscillator agent, the state of which resides on a unit circle."""

    def __init__(self, simulator, k, mu, nat_freq, f0, init_state=None, state_dim=1):
        """
        Initialize an oscillator agent.

        Args:
            simulator: The simulator in which the agent is initialized
            k: Sensitivity to local control
            mu: Sensitivity to cluster control
            nat_freq: Natural frequency of an oscillator
            f0: Frequency for a sine controller
            init_state: Initial state
            state_dim: Dimensionality of state space
        """

        self.__k = k
        self.__mu = mu
        self.__local_controller = KuramotoController(simulator, self, nat_freq)
        self.__cluster_controller = SineController(simulator, f0)
        if init_state:
            self.__state = init_state
        else:
            self.__state = 2 * np.pi * np.random.rand(state_dim)[0]

    def send_output(self):
        """Get agent output."""
        return self.__state

    def update_state(self, do_cluster_control):
        """
        Update agent state.

        Args:
            do_cluster_control: True if apply cluster control
        """
        if do_cluster_control:
            cc_out = self.__mu * self.__cluster_controller.get_control_input()
        else:
            cc_out = 0
        lc_out = self.__k * self.__local_controller.get_control_input()
        self.__state += lc_out + cc_out

    def change_local_controller(self, new_local_controller):
        """
        Change local control input

        Args:
            new_local_controller: New local controller for replacement
        """
        self.__local_controller = new_local_controller 

    def change_cluster_controller(self, new_cluster_controller): 
        """
        Change cluster control input

        Args:
            new_cluster_controller: New cluster controller for replacement
        """
        self.__cluster_controller = new_cluster_controller
