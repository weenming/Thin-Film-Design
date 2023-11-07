import torch


def E_to_R(spectrum, s_ratio=1, p_ratio=1):
    '''E: 2D TORCH tensor, (wls.shape[0] * 2, 2) (s+, p+ // s-, p-)
    '''
    wls_size = spectrum.shape[0] // 2
    
    rs = spectrum[:wls_size, 1] / spectrum[:wls_size, 0]
    rp = spectrum[wls_size:, 1] / spectrum[wls_size:, 0]

    # Rs = (s_ratio * rs.conj() * rs).real
    # Rp = (p_ratio * rp.conj() * rp).real
    Rs = s_ratio * rs.abs().square()
    Rp = p_ratio * rp.abs().square()

    return (Rs + Rp) / (s_ratio + p_ratio)


def E_to_tan2Psi(spectrum):
    '''
    E: 2D TORCH tensor, (wls.shape[0] * 2, 2) (s+, p+ // s-, p-)
    returns tan^2 \Psi
    '''
    wls_size = spectrum.shape[0] // 2
    
    rs =  spectrum[:wls_size, 1] / spectrum[:wls_size, 0]
    rp =  spectrum[wls_size:, 1] / spectrum[wls_size:, 0]

    Rs = (rs.conj() * rs).real
    Rp = (rp.conj() * rp).real
    
    return Rp / Rs


def E_to_phase(spectrum):
    '''
    E: 2D TORCH tensor, (wls.shape[0] * 2, 2) (s+, p+ // s-, p-)
    returns tan^2 \Psi
    '''
    wls_size = spectrum.shape[0] // 2
    
    rs =  spectrum[:wls_size, 1] / spectrum[:wls_size, 0]
    rp =  spectrum[wls_size:, 1] / spectrum[wls_size:, 0]

    Rs = (rs.conj() * rs).real
    Rp = (rp.conj() * rp).real
    
    norm = (Rp / Rs).sqrt()
    return rp / rs / norm
    