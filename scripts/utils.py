import math
import numcodecs
import numpy as np


def orientation_ensure_3d(orientation):
    """
    Parameters
    ----------
    orientation : str
        Either one of {'coronal', 'axial', 'sagittal'}, or a two- or
        three-letter permutation of {('R', 'L'), ('A', 'P'), ('S', 'I')}

    Returns
    -------
    orientation : str
        A three-letter permutation of {('R', 'L'), ('A', 'P'), ('S', 'I')}
    """
    orientation = {
        'coronal': 'LI',
        'axial': 'LP',
        'sagittal': 'PI',
    }.get(orientation.lower(), orientation).upper()
    if len(orientation) == 2:
        if 'L' not in orientation and 'R' not in orientation:
            orientation += 'R'
        if 'P' not in orientation and 'A' not in orientation:
            orientation += 'A'
        if 'I' not in orientation and 'S' not in orientation:
            orientation += 'S'
    return orientation


def orientation_to_affine(orientation, vxw=1, vxh=1, vxd=1):
    orientation = orientation_ensure_3d(orientation)
    affine = np.zeros([4, 4])
    vx = np.asarray([vxw, vxh, vxd])
    for i in range(3):
        letter = orientation[i]
        sign = -1 if letter in 'LPI' else 1
        letter = {'L': 'R', 'P': 'A', 'I': 'S'}.get(letter, letter)
        index = list('RAS').index(letter)
        affine[index, i] = sign * vx[i]
    return affine


def center_affine(affine, shape):
    if len(shape) == 2:
        shape = [*shape, 1]
    shape = np.asarray(shape)
    affine[:3, -1] = -0.5 * affine[:3, :3] @ (shape - 1)
    return affine


def ceildiv(x, y):
    return int(math.ceil(x / y))


def floordiv(x, y):
    return int(math.floor(x / y))


def make_compressor(name, **prm):
    if not isinstance(name, str):
        return name
    name = name.lower()
    if name == 'blosc':
        Compressor = numcodecs.Blosc
    elif name == 'zlib':
        Compressor = numcodecs.Zlib
    else:
        raise ValueError('Unknown compressor', name)
    return Compressor(**prm)


ome_valid_units = {
    'space': [
        'angstrom',
        'attometer',
        'centimeter',
        'decimeter',
        'exameter',
        'femtometer',
        'foot',
        'gigameter',
        'hectometer',
        'inch',
        'kilometer',
        'megameter',
        'meter',
        'micrometer',
        'mile',
        'millimeter',
        'nanometer',
        'parsec',
        'petameter',
        'picometer',
        'terameter',
        'yard',
        'yoctometer',
        'yottameter',
        'zeptometer',
        'zettameter',
    ],
    'time': [
        'attosecond',
        'centisecond',
        'day',
        'decisecond',
        'exasecond',
        'femtosecond',
        'gigasecond',
        'hectosecond',
        'hour',
        'kilosecond',
        'megasecond',
        'microsecond',
        'millisecond',
        'minute',
        'nanosecond',
        'petasecond',
        'picosecond',
        'second',
        'terasecond',
        'yoctosecond',
        'yottasecond',
        'zeptosecond',
        'zettasecond',
    ]
}

nifti_valid_units = [
    'unknown',
    'meter',
    'mm',
    'micron',
    'sec',
    'msec',
    'usec',
    'hz',
    'ppm',
    'rads',
]

si_prefix_short2long = {
    'Q': 'quetta',
    'R': 'ronna',
    'Y': 'yotta',
    'Z': 'zetta',
    'E': 'exa',
    'P': 'peta',
    'T': 'tera',
    'G': 'giga',
    'M': 'mega',
    'K': 'kilo',
    'k': 'kilo',
    'H': 'hecto',
    'h': 'hecto',
    'D': 'deca',
    'da': 'deca',
    'd': 'deci',
    'c': 'centi',
    'm': 'milli',
    'u': 'micro',
    'μ': 'micro',
    'n': 'nano',
    'p': 'pico',
    'f': 'femto',
    'a': 'atto',
    'z': 'zepto',
    'y': 'yocto',
    'r': 'ronto',
    'q': 'quecto',
}

si_prefix_long2short = {
    long: short
    for short, long in si_prefix_short2long.items()
}


si_prefix_exponent = {
    'Q': 30,
    'R': 27,
    'Y': 24,
    'Z': 21,
    'E': 18,
    'P': 15,
    'T': 12,
    'G': 9,
    'M': 6,
    'K': 3,
    'k': 3,
    'H': 2,
    'h': 2,
    'D': 1,
    'da': 1,
    '': 0,
    'd': -1,
    'c': -2,
    'm': -3,
    'u': -6,
    'μ': -6,
    'n': -9,
    'p': -12,
    'f': -15,
    'a': -18,
    'z': -21,
    'y': -24,
    'r': -27,
    'q': -30,
}


unit_space_short2long = {
    short + 'm': long + 'meter'
    for short, long in si_prefix_long2short
}
unit_space_short2long.update({
    'm': 'meter',
    'mi': 'mile',
    'yd': 'yard',
    'ft': 'foot',
    'in': 'inch',
    "'": 'foot',
    '"': 'inch',
    'Å': 'angstrom',
    'pc': 'parsec',
})
unit_space_long2short = {
    long: short
    for short, long in unit_space_short2long.items()
}
unit_space_long2short['micron'] = 'u'

unit_time_short2long = {
    short + 's': long + 'second'
    for short, long in si_prefix_long2short
}
unit_time_short2long.update({
    'y': 'year',
    'd': 'day',
    'h': 'hour',
    'm': 'minute',
    's': 'second',
})
unit_time_long2short = {
    long: short
    for short, long in unit_time_short2long.items()
}

unit_space_scale = {
    prefix + 'm': 10**exponent
    for prefix, exponent in si_prefix_exponent
}
unit_space_scale.update({
    'mi': 1609.344,
    'yd': 0.9144,
    'ft': 0.3048,
    "'": 0.3048,
    'in': 25.4E-3,
    '"': 25.4E-3,
    'Å': 1E-10,
    'pc': 3.0857E16,
})

unit_time_scale = {
    prefix + 's': 10**exponent
    for prefix, exponent in si_prefix_exponent
}
unit_time_scale.update({
    'y': 365.25*24*60*60,
    'd': 24*60*60,
    'h': 60*60,
    'm': 60,
})


def convert_unit(value, src, dst):
    src = unit_to_scale(src)
    dst = unit_to_scale(dst)
    return value * (src / dst)


def to_ome_unit(unit):
    if unit in unit_space_short2long:
        unit = unit_space_short2long[unit]
    elif unit in unit_time_short2long:
        unit = unit_time_short2long[unit]
    elif unit in si_prefix_short2long:
        unit = si_prefix_short2long[unit]
    if unit not in (*ome_valid_units['space'], *ome_valid_units['time']):
        raise ValueError('Unknow unit')


def to_nifti_unit(unit):
    unit = to_ome_unit(unit)
    return {
        'meter': 'meter',
        'millimeter': 'mm',
        'micrometer': 'micron',
        'second': 'sec',
        'millisecond': 'msec',
        'microsecond': 'usec',
    }.get(unit, 'unknown')


def unit_to_scale(unit):
    if unit in unit_space_long2short:
        unit = unit_space_long2short[unit]
    elif unit in unit_time_long2short:
        unit = unit_time_long2short[unit]
    elif unit in si_prefix_long2short:
        unit = si_prefix_long2short[unit]
    if unit in unit_space_scale:
        unit = unit_space_scale[unit]
    elif unit in unit_time_scale:
        unit = unit_time_scale[unit]
    elif unit in si_prefix_exponent:
        unit = 10 ** si_prefix_exponent[unit]
    if isinstance(unit, str):
        raise ValueError('Unknown unit', unit)
    return unit
