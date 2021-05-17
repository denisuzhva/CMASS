# sensor.py


import numpy as np
from numpy.core.numeric import full


class Sensor:
    """A sensor for multiagen system remote compressied observations."""

    def __init__(self, n_pixels, exposure, ftop, fbot, sim_dt, meas_matrix, m, record_full=True):
        """
        Initialize a sensor.

        Args:
            n_pixels: Number of pixels
            exposure: Exposure time
            ftop: Upper limit of field of view
            fbot: Bottom limit of field of view
            sim_dt: Simulation time step
            meas_matrix: Measurement matrix for compression
            m: Dimensionality of compressed data
            record_full: True if record non-compressed data
        """
        self.__n_pixels = n_pixels
        self.__pixel_ticks = np.linspace(fbot, ftop, self.__n_pixels+1)
        self.__sample_iter = 0
        self.__samples_per_shot = int(exposure // sim_dt) 
        self.__buffer = []
        self.__m = m
        self.__meas_matrix = meas_matrix 
        self.__cs_recording = np.zeros((m, 1)) # recorder for compressed data
        self.__record_full = record_full 
        if record_full:
            self.__full_recording = np.zeros((self.__n_pixels, 1)) # recorder for non-compressed data
        else:
            self.__full_recording = None

    def accept_sample(self, sample):
        """
        Accept pure agent states at some iteration.

        Args: 
            sample: States of the agents
        """
        # Increase sensor iteration
        self.__sample_iter += 1

        # Write a sample by filling the buffer
        for item in sample:
            self.__buffer.append(item)

        # Simulate observation when exposure time is passed
        if self.__sample_iter >= self.__samples_per_shot:
            self.__sample_iter = 0 # reset sensor iteration
            rec, _ = np.histogram(self.__buffer, density=True, bins=self.__pixel_ticks) # simulate observation from the buffer
            self.__buffer = [] # clear buffer
            rec[rec > 0] = 1. # thresholding
            cs_rec = self.__meas_matrix.dot(rec) # compression
            self.__cs_recording[:, -1] = cs_rec # compressed data recording
            self.__cs_recording = np.append(self.__cs_recording, np.zeros((self.__m, 1)), axis=1) # expand recorder for future observations

            # Record non-compressed data if needed
            if self.__record_full:
                self.__full_recording[:, -1] = rec
                self.__full_recording = np.append(self.__full_recording, np.zeros((self.__n_pixels, 1)), axis=1)

    def reset_recordings(self):
        """Clear the recorder data."""
        self.__cs_recording = np.zeros((self.__m, 1))
        if self.__record_full:
            self.__full_recording = np.zeros((self.__n_pixels, 1))

    def get_recording(self):
        """
        Get the recorder data.

        Returns:
            self.__cs_recording: Compressed recording data 
            self.__full_recording: Non-compressed recording data
        """
        return self.__cs_recording, self.__full_recording




