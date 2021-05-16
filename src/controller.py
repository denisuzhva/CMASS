# controller.py


from abc import ABCMeta, abstractmethod
import numpy as np


class Controller(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_control_input(self):
        pass


class LocalController(Controller):

    def __init__(self, simulator, master_agent, nat_freq):
        self.__simulator = simulator
        self.__master_agent = master_agent
        self.__nat_freq = nat_freq

    def get_control_input(self):
        neighbor_outputs = self.__simulator.get_neighbor_outputs(self.__master_agent)
        master_output = self.__master_agent.send_output()
        sines = [np.sin(neighbor_output - master_output) for neighbor_output in neighbor_outputs]
        control_input = np.sum(np.array(sines)) + self.__nat_freq
        control_input *= self.__simulator.get_dt()
        return control_input


class ClusterController(Controller):

    def __init__(self, simulator, f0):
        self.__simulator = simulator
        self.__f0 = f0

    def get_control_input(self):
        control_input = np.sin(2 * self.__f0 * np.pi * self.__simulator.get_time()) * self.__simulator.get_dt()
        return control_input