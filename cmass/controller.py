# controller.py


from abc import ABCMeta, abstractmethod
import numpy as np


class Controller(metaclass=ABCMeta):
    """An interface for agent controller. All the methods below must be realized in a child class."""
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_control_input(self):
        pass


class KuramotoController(Controller):
    """A controller which implements local sinusoidal Kuramoto protocol."""

    def __init__(self, simulator, master_agent, nat_freq):
        """
        Initialize a controller.

        Args:
            simulator: The simulator in which the agent is initialized
            master_agent: An agent in which the controller is initialized
            nat_freq: Natural frequency of the master agent
        """
        self.__simulator = simulator
        self.__master_agent = master_agent
        self.__nat_freq = nat_freq

    def get_control_input(self):
        """
        Get control input based on master agent state and states of its neighbors.

        Returns:
            control input: Calculated control input
        """
        # Get outputs
        neighbor_outputs = self.__simulator.get_neighbor_outputs(self.__master_agent)
        master_output = self.__master_agent.send_output()

        # Calculate control input
        sines = [np.sin(neighbor_output - master_output) for neighbor_output in neighbor_outputs]
        control_input = np.sum(np.array(sines)) + self.__nat_freq
        control_input *= self.__simulator.get_dt()
        return control_input


class SineController(Controller):
    """A simple sinusoidal controller without a feedback."""

    def __init__(self, simulator, f0):
        """
        Initialize a controller.

        Args:
            simulator: The simulator in which the agent is initialized
            f0: Sine frequency
        """
        self.__simulator = simulator
        self.__f0 = f0

    def get_control_input(self):
        """
        Get control input based on current time and the specified frequency.

        Returns:
            control_input: Calculated control input
        """
        control_input = np.sin(2 * self.__f0 * np.pi * self.__simulator.get_time()) * self.__simulator.get_dt()
        return control_input