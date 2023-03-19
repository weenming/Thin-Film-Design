import numpy as np
import film
import gets.get_spectrum as get_spectrum


class BaseSpectrum:
    def __init__(self):
        pass

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

class Spectrum(BaseSpectrum):
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
    def __init__(self, incident_angle, wavelengths, spec_R, spec_T=None):
        self.INC_ANG = incident_angle
        self.WLS = wavelengths
        self.spec_R = spec_R
        if spec_T is None:
            self.spec_T = 1 - spec_R
        else:
            self.spec_T = spec_T
    

class SpectrumSimple(BaseSpectrum):
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

    def update_n(self):
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

    def is_updated(self):
        return self.updated