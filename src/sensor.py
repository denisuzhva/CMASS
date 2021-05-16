# sensor.py


import numpy as np
from numpy.core.numeric import full


class Sensor:

    def __init__(self, n_pixels, exposure, ftop, fbot, sim_dt, meas_matrix, m, record_full=True):
        self.__n_pixels = n_pixels
        self.__pixel_ticks = np.linspace(fbot, ftop, self.__n_pixels+1)
        self.__sample_iter = 0
        self.__samples_per_shot = int(exposure // sim_dt)
        self.__buffer = []

        self.__m = m
        self.__meas_matrix = meas_matrix 
        self.__cs_recording = np.zeros((m, 1))

        self.__record_full = record_full 
        if record_full:
            self.__full_recording = np.zeros((self.__n_pixels, 1))
        else:
            self.__full_recording = None

    def accept_sample(self, sample):
        self.__sample_iter += 1
        for item in sample:
            self.__buffer.append(item)
        if self.__sample_iter >= self.__samples_per_shot:
            self.__sample_iter = 0
            rec, _ = np.histogram(self.__buffer, density=True, bins=self.__pixel_ticks)
            self.__buffer = []
            rec[rec > 0] = 1.
            cs_rec = self.__meas_matrix.dot(rec)
            self.__cs_recording[:, -1] = cs_rec
            self.__cs_recording = np.append(self.__cs_recording, np.zeros((self.__m, 1)), axis=1)
            if self.__record_full:
                self.__full_recording[:, -1] = rec
                self.__full_recording = np.append(self.__full_recording, np.zeros((self.__n_pixels, 1)), axis=1)

    def reset_recordings(self):
        self.__cs_recording = np.zeros((self.__m, 1))
        if self.__record_full:
            self.__full_recording = np.zeros((self.__n_pixels, 1))

    def get_recording(self):
        return self.__cs_recording, self.__full_recording




