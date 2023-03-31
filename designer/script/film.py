import numpy as np
import copy
import gets.get_n as get_n
import spectrum


class Film:
    def __init__(self):
        self.d = np.array([])
        self.n = np.array([], dtype='complex')


class FilmSimple(Film):
    """
    Film that consist of 2 materials. Only d is modifiable. Materials are
    non - absorbing。

    Structure of the film: ABAB...substrate.
    0 - th layer is incident material.
    1 - st layer (self.d[0]) is A
    2 - nd layer (self.d[1]) is B
    ...
    last layer is substrate

    Arguments:
        A, B, substrate(str):
            material name of A, B, substrate

        d_init(numpy array): initial d.
        incidence(str): material of incidence

    Attributes:
        d(numpy array):
            thicknesses of each layer. d[0] is the first A.
            d.shape[0] is the layer number
        get_n_A, get_n_B, get_n_sub, get_n_inc (function: wl -> n):
            refractive indices of materials A, B, substrate and
            incident material
        spectrum(list of SpectrumSimple instances):

        n_arr (numpy array):
            array of refractive indices of layers at different wls


    """

    def __init__(self, A:str, B:str, substrate:str, d_init: np.array, incidence='Air'):
        try:
            exec(f"self.get_n_A = get_n.get_n_{A}")
            exec(f"self.get_n_B = get_n.get_n_{B}")
            exec(f"self.get_n_sub = get_n.get_n_{substrate}")
            exec(f"self.get_n_inc = get_n.get_n_{incidence}")
        except:
            raise ValueError(
                "Bad material. Dispersion must be specified in gets.get_n")
        
        if d_init.shape == ():
            d_init = np.array([d_init])

        assert len(d_init.shape) == 1, "Should be 1 dim array!"
        assert d_init.shape[0] < 250, "Too many layers!"
    
        self.d = d_init
        
        self.spectrum = []

    # Getter and Setter, etc of attribute spectrum
    def add_spec_param(self, inc_ang, wls):
        """
        Setter of the spectrum params: wls and inc

        """
        for s in self.spectrum:
            if np.array_equal(s.WLS, wls) and s.INC_ANG == inc_ang:
                return
        spec = spectrum.SpectrumSimple(inc_ang, wls, self)
        self.spectrum.append(spec)
        return spec

    def remove_spec_param(self, inc_ang=None, wls=None):
        assert wls is not None or inc_ang is not None, "Must specify which spec to del"
        count = 0
        for s in self.spectrum:
            if (inc_ang is None and np.array_equal(s.WLS, wls)) or \
                (wls is None and s.INC_ANG == inc_ang) or \
                (np.array_equal(s.WLS, wls) == s.INC_ANG == inc_ang):
                self.spectrum.remove(s)
                count += 1
        return count

    def get_spec(self, inc_ang=None, wls=None) -> spectrum.SpectrumSimple:
        """ return spectrum with specified wls and inc_ang
        """
        if len(self.spectrum) == 1 and inc_ang is None and wls is None:
            # when only one spectrum, return the only one spectrum
            return self.spectrum[0]

        else:
            if inc_ang is None or wls is None:
                raise ValueError(
                    "In the case of multiple spectrums, must specify inc_ang\
                    and wls")
            for s in self.spectrum:
                if np.array_equal(s.WLS, wls) and s.INC_ANG == inc_ang:
                    return s
            # Not found, add to get_spec
            # print("WARNING: spec not in this film's spec list. New spec added to film!")
            return self.add_spec_param(inc_ang, wls)


    def get_all_spec_list(self) -> list[spectrum.SpectrumSimple]:
        return self.spectrum

    def calculate_spectrum(self):
        for s in self.spectrum:
            s.calculate()

    # all layers should have non-zero thickness, except right after insertion
    # so check by explicitly calling these methods
    def check_thickness(self):
        assert np.min(self.d) > 0, "layers of zero thickness!"

    def remove_negative_thickness_layer(self, exclude=[]):
        indices = []
        d = self.get_d()
        # first layer is never removed
        i = 1
        while i < d.shape[0] - 1:
            if d[i] <= 0 and i not in exclude:
                d[i - 1] += d[i + 1]
                d = np.delete(d, [i, i + 1])
                exclude = [x - 2 for x in exclude]
            else:
                i += 1

        if d[-1] <= 0 and d.shape[0] not in exclude:
            d = np.delete(d, -1)

        self.update_d(d)

    # Helper functions of insertion
    def insert_layer(self, layer_index, position, thickness):
        """
        insert a layer at the specified position
            B  ||   A   ||  B
        insert at:  ^    (i == layer_index)
            B  ||A||B||A||  B
                 ^       (i == layer_index)
                    ^    (i == layer_index + 1)
                       ^ (i == layer_index + 2)
        """
        d = self.get_d()
        assert 0 <= layer_index < d.shape[0], 'invalid insert layer'
        
        assert d[layer_index] >= position - 1e-5 and position >= 0, \
            'invalid insert position'

        d = np.insert(d, [layer_index + 1, layer_index + 1], thickness)
        d[layer_index + 2] = max(d[layer_index] - position, 0)
        d[layer_index] = position
        self.update_d(d)


    def get_insert_layer_n(self, index):
        """
        Insert layer should have a different material.
        e.g. ABABA, the material to insert at the 3-rd layer is: A
        """
        assert index < self.d.shape[0]
        if index % 2 == 0:
            return self.A
        else:
            return self.B


    # Functions to calculate n at given wls
    def calculate_n_array(self, wls: np.array):
        """
        calculate n at different wl for each layer.

        Returns:
            2d np.array, size is wls number * layer number. Refractive indices
        """
        n_arr = np.empty((wls.shape[0], self.get_layer_number()), dtype='complex128')

        n_A: np.array = self.get_n_A(wls)
        n_B: np.array = self.get_n_B(wls)
        l = self.get_layer_number()
        
        n_arr[:, [i for i in range(0, l, 2)]] = n_A.reshape((-1, 1))
        n_arr[:, [i for i in range(1, l, 2)]] = n_B.reshape((-1, 1))
        return n_arr


    def calculate_n_sub(self, wls):
        """
        calculate n at different wl for substrate

        Returns:
            1d np.array, size is wls.shape[0]. Refractive indices
        """
        n_arr = np.empty(wls.shape[0], dtype='complex128')
        for i in range(wls.shape[0]):
            n_arr[i] = self.get_n_sub(wls[i])
        return n_arr

    def calculate_n_inc(self, wls):
        """
        calculate n at different wl for incident material

        Returns:
            1d np.array, size is wls.shape[0]. Refractive indices

        """
        n_arr = np.empty(wls.shape[0], dtype='complex128')
        for i in range(wls.shape[0]):
            n_arr[i] = self.get_n_inc(wls[i])
        return n_arr


    # Accessor functions
    def get_d(self):
        return self.d

    def update_d(self, d):
        self.d = d
        for spec in self.get_all_spec_list():
            spec.update_n() # calculate the n array again
            spec.outdate() # set the oudated flag of the stored spectrum(s)
        
        return

    def get_layer_number(self):
        return self.d.shape[0]


    # Other utility functions
    def get_optical_thickness(self, wl, neglect_last_layer=False) -> float:
        """
        Calculate the optical thickness of this film

        Arguments:
            wl (float):
                wavelength at which refractive index is evaluated
        """
        l = self.get_layer_number()
        n_A = self.get_n_A(wl)
        n_B = self.get_n_B(wl)
        d_even = np.array([self.get_d()[i] for i in range(0, l, 2)])
        d_odd = np.array([self.get_d()[i] for i in range(1, l, 2)])
        ot = np.sum(n_A * d_even) + np.sum(n_B * d_odd)            
        if neglect_last_layer:
            ot -= self.get_d()[-1] * (n_A if l % 2 == 1 else n_B)
        return ot


    def remove_thin_layers(self, d_min, method):

        f_sub = copy.deepcopy(self)
        method(f_sub, d_min)
        return f_sub

