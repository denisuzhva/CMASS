# agent.py


from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.core.numeric import full
from controller import *


class Agent(metaclass=ABCMeta):

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
    def change_local_strategy(self, local_parameters):
        pass

    @abstractmethod
    def change_cluster_strategy(self, cluster_parameters):
        pass


class OscillatorAgent(Agent):

    def __init__(self, simulator, k, nat_freq, mu, f0, init_state=None, state_dim=1):

        self.__k = k
        self.__mu = mu
        self.__local_controller = LocalController(simulator, self, nat_freq)
        self.__cluster_controller = ClusterController(simulator, f0)
        if init_state:
            self.__state = init_state
        else:
            self.__state = 2 * np.pi * np.random.rand(state_dim)[0]

    def send_output(self):
        return self.__state

    def update_state(self, do_cluster_control):
        if do_cluster_control:
            cc_out = self.__mu * self.__cluster_controller.get_control_input()
        else:
            cc_out = 0
        lc_out = self.__k * self.__local_controller.get_control_input()
        self.__state += lc_out + cc_out

    def change_local_strategy(self, local_controller):
        self.__local_controller = local_controller 

    def change_cluster_strategy(self, cluster_controller): 
        self.__cluster_controller = cluster_controller
