import numpy as np
from numpy.typing import NDArray
import copy
import utils.get_n as get_n
from spectrum import SpectrumSimple
from abc import ABC, abstractmethod
from typing import Callable
import tmm.get_spectrum as get_spectrum


class BaseFilm(ABC):
    d: NDArray
    spectrums: list[SpectrumSimple]

    def __init__(self, substrate, incidence):
        self.materials = {}
        self._register_get_n('sub', substrate)
        self._register_get_n('inc', incidence)

    def _register_get_n(self, name: str, material):
        if type(material) is str:
            try:
                exec(f"self.get_n_{name} = get_n.get_n_{material}")
                self.materials[name] = material
            except:
                raise ValueError(
                    "Material not found. \
                    Dispersion must have been defined in gets.get_n")
        elif type(float(material)) is float:
            exec(
                f"self.get_n_{name} = lambda wl: wl * {material} / wl")
            self.materials[name] = material
        else:
            raise ValueError(
                'bad material. should be either name defined in utils.get_n or a float')

    # spectrum-related methods

    def add_spec_param(self, inc_ang, wls):
        """
        Setter of the spectrum params: wls and inc

        """

        for s in self.spectrums:
            if np.array_equal(s.WLS, wls) and s.INC_ANG == inc_ang:
                return s
        spec = SpectrumSimple(inc_ang, wls, self)
        self.spectrums.append(spec)
        return spec

    def remove_spec_param(self, inc_ang=None, wls=None):
        assert wls is not None or inc_ang is not None, "Must specify which spec to del"
        count = 0
        for s in self.spectrums:
            if (inc_ang is None and np.array_equal(s.WLS, wls)) or \
                (wls is None and s.INC_ANG == inc_ang) or \
                    (np.array_equal(s.WLS, wls) == s.INC_ANG == inc_ang):
                self.spectrums.remove(s)
                count += 1
        return count

    def remove_all_spec_param(self):
        self.spectrums = []
        return

    def get_spec(self, inc_ang=None, wls=None) -> SpectrumSimple:
        """ return spectrum with specified wls and inc_ang
        """
        if len(self.spectrums) == 1 and inc_ang is None and wls is None:
            # when only one spectrum, return the only one spectrum
            return self.spectrums[0]

        else:
            if inc_ang is None or wls is None:
                raise ValueError(
                    "In the case of multiple spectrums, must specify inc_ang\
                    and wls")
            for s in self.spectrums:
                if np.array_equal(s.WLS, wls) and s.INC_ANG == inc_ang:
                    return s
            # Not found, add to get_spec
            # print("WARNING: spec not in this film's spec list. New spec added to film!")
            return self.add_spec_param(inc_ang, wls)

    def get_all_spec_list(self) -> list[SpectrumSimple]:
        return self.spectrums

    # n_array related
    @abstractmethod
    def calculate_n_array(self, wls: NDArray):
        raise NotImplementedError

    def calculate_n_sub(self, wls):
        """
        calculate n at different wl for substrate

        Returns:
            1d NDArray, size is wls.shape[0]. Refractive indices
        """
        n_arr = np.empty(wls.shape[0], dtype='complex128')
        for i in range(wls.shape[0]):
            n_arr[i] = self.get_n_sub(wls[i])
        return n_arr

    def calculate_n_inc(self, wls):
        """
        calculate n at different wl for incident material

        Returns:
            1d NDArray, size is wls.shape[0]. Refractive indices

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
            spec.outdate()  # set the oudated flag of the stored spectrum(s)
        return

    def get_layer_number(self):
        return self.d.shape[0]

    @abstractmethod
    def get_optical_thickness(self, wl, neglect_last_layer=False) -> float:
        raise NotImplementedError

    @abstractmethod
    def calculate_spectrum(self):
        raise NotImplementedError


class FreeFormFilm(BaseFilm):
    def __init__(
        self,
        init_n_ls: NDArray,
        total_gt,
        substrate: str,
        incidence='Air',
        allowed_materials=None
    ):
        '''
            Specify a new FreeFormFilm.
            NOTE: zero dispersion is assumed

            Parameters:
                init_n_arr:
                    Every layers' refractive indix is relaxed to evolve 
                    continuously. No dispersion supported currently
                total_gt:
                    total allowed geometric thickness. As described in \cite{}
                    total thickness of a layer is a key constraint to the lower
                    bound of the design loss.
                    Layer thickness is determined based on an even subdivision
                    w.r.t. geometrical thickness.
                allowed_materials:
                    A discrete set of materials that is allowed. One possible
                    strategy to include with this constraint is a projection 
                    after the optimization is complete.


        '''
        super().__init__(substrate, incidence)  # register sub and inc

        if allowed_materials is not None:
            raise NotImplementedError
        self.d = np.ones(init_n_ls.shape[0], dtype='float')
        self.d *= total_gt / (self.d).sum()
        init_n_ls = init_n_ls.astype('complex128')
        self.n = init_n_ls
        self.spectrums = []

    def calculate_n_array(self, wls: NDArray):
        n_arr = np.empty((wls.shape[0], self.get_layer_number()),
                         dtype='complex128')
        for i in range(self.get_layer_number()):
            n_arr[:, i] = get_n.get_n_free(wls, self.n[i])
        return n_arr

    def get_optical_thickness(self, wl, neglect_last_layer=False) -> float:
        if neglect_last_layer:
            raise NotImplementedError
        return (self.d * self.calculate_n_array(np.array([wl]))[0, :].real).sum()

    def update_n(self, n_new):
        self.n = n_new
        for s in self.get_all_spec_list():
            s.outdate()

    def get_n(self):
        '''
            Returns 1-d array of refractive indices.

            Because FreeFormFilm contains only non-dispersive materials IN the 
            film stack (substrate and incidence can be dispersive though)
            the n_array is duplicate along axis 0. 
        '''
        return self.n

    def calculate_spectrum(self):
        for s in self.spectrums:
            s.calculate(get_spectrum.get_spectrum_free)

    def project_to_two_material_film(self, n1, n2, material1=None, material2=None):
        if n1 < n2:  # assume n1 > n2
            n1, n2 = n2, n1
            material1, material2 = material2, material1

        is_high = (self.get_n() > ((n1 + n2) / 2))
        each_d = self.get_d()[0]
        new_d = np.array([0])  # assume first layer: high refractive index (n1)
        now_high = True
        for i in range(self.get_layer_number()):
            if (is_high[i] and now_high) or (not is_high[i] and not now_high):
                new_d[-1] += each_d
            else:
                now_high = not now_high
                new_d = np.append(new_d, each_d)
        if material1 is not None and material2 is not None:
            new_film = TwoMaterialFilm(
                material1, material2, self.materials['sub'], new_d)
        else:
            new_film = TwoMaterialFilm(n1, n2, self.materials['sub'], new_d)

        return new_film


class TwoMaterialFilm(BaseFilm):
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
    get_n_A: Callable
    get_n_B: Callable

    def __init__(
        self,
        A: str,
        B: str,
        substrate: str,
        d_init: NDArray,
        incidence='Air'
    ):
        super().__init__(substrate, incidence)  # register sub and inc
        self._register_get_n('A', A)
        self._register_get_n('B', B)

        if d_init.shape == ():
            d_init = np.array([d_init])

        assert len(d_init.shape) == 1, "Should be 1 dim array!"

        self.d = d_init
        self.spectrums: list[SpectrumSimple] = []

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

    def remove_thin_layers(self, d_min, method):

        f_sub = copy.deepcopy(self)
        method(f_sub, d_min)
        return f_sub

    # Helper functions of needle insertion

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

    # Implement abstract methods

    def calculate_n_array(self, wls: NDArray):
        """
        calculate n at different wl for each layer.

        Returns:
            2d NDArray, size is wls number * layer number. Refractive indices
        """
        n_arr = np.empty(
            (wls.shape[0], self.get_layer_number()), dtype='complex128')

        n_A: NDArray = self.get_n_A(wls)
        n_B: NDArray = self.get_n_B(wls)
        l = self.get_layer_number()

        n_arr[:, [i for i in range(0, l, 2)]] = n_A.reshape((-1, 1))
        n_arr[:, [i for i in range(1, l, 2)]] = n_B.reshape((-1, 1))
        return n_arr

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

    def calculate_spectrum(self):
        for s in self.spectrums:
            s.calculate(get_spectrum.get_spectrum_simple)


class EqOTFilm(FreeFormFilm):
    '''
    Free Form film, but constrain \tau_i same instead of di same.
    '''

    def __init__(
        self,
        init_n_ls: NDArray,
        total_ot,
        substrate: str,
        incidence='Air',
        allowed_materials=None
    ):
        WAHTEVER_WL = 1000

        super().__init__(
            init_n_ls,
            total_ot,
            substrate,
            incidence,
            allowed_materials
        )  # register sub and inc

        self.d /= self.get_n().real
        self.d *= total_ot / \
            self.get_optical_thickness(WAHTEVER_WL)
