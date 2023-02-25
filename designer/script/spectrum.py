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

    def __init__(self, incident_angle, wavelengths, film):
        self.INC_ANG = incident_angle
        self.WLS = wavelengths
        self.n = film.calculate_n_array(self.WLS)
        self.n_sub = film.calculate_n_sub(self.WLS)
        self.n_inc = film.calculate_n_inc(self.WLS)
        self.spec = np.empty(self.WLS.shape[0] * 2)
        self.film = film
        self.updated = False

    def set_n(self):
        # [R, T]
        self.n = self.film.calculate_n_array(self.WLS) 

    def calculate(self):
        # only R spectrum
        get_spectrum.get_spectrum_simple(self.spec, self.WLS,
                                                self.film.d,
                                                self.n,
                                                self.n_sub,
                                                self.n_inc,
                                                self.INC_ANG
                                                )
        self.spec_R = self.spec[:self.WLS.shape[0]]
        self.spec_T = self.spec[self.WLS.shape[0]:]
        self.updated = True

    def outdate(self):
        self.updated = False

    def get_R(self):
        try:
            return self.spec_R
        except AttributeError:
            raise ValueError('spec R not yet calculated')

    
    def get_T(self):
        try:
            return self.spec_T
        except AttributeError:
            raise ValueError('spec R not yet calculated')

        

    def is_updated(self):
        return self.updated