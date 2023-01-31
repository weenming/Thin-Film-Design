import numpy as np
import film
import gets.get_spectrum as get_spectrum


class Spectrum:
    """
    An object containing the spectrum

    Attributes:
        INC_ANG (float):
        WLS (np.array):
        spec_R (np.array):
            The REFLECTANCE spectrum.
        spec_T (np.array):
            The transmittance spectrum. Subject to future changes
    """

    def __init__(self, incident_angles, wavelengths):
        pass


class SpectrumSimple(Spectrum):
    """
    An object containing the spectrum

    Attributes:    
    INC_ANG (float):
    WLS (np.array):
    spec (np.array):
        The REFLECTANCE spectrum. In FilmSimple instances, no absorption so 
        only R is considered
    updated:
        When an instance of Film has a SpectrumSimple, update denotes whether 
        this spectrum is updated in regard to the current film structure.
    """

    def __init__(self, incident_angles, wavelengths, film):
        self.INC_ANG = incident_angles
        self.WLS = wavelengths
        self.n = film.calculate_n_array(self.WLS)
        self.n_sub = film.calculate_n_sub(self.WLS)
        self.n_inc = film.calculate_n_inc(self.WLS)
        self.spec = None
        self.film = film
        self.updated = False

    def set_n(self):
        self.n = self.calculate_n_array(self.WLS)

    def calculate(self):
        # only R spectrum
        self.spec = get_spectrum.get_spectrum_R(self.WLS,
                                                self.film.d,
                                                self.n,
                                                self.n_sub,
                                                self.n_inc,
                                                theta=self.INC_ANG
                                                )
        self.spec_R = self.spec[0, :]
        self.spec_T = self.spec[1, :]
        self.updated = True

    def outdate(self):
        self.updated = False

    def get_R(self):
        return self.spec_R

    def is_updated(self):
        return self.updated