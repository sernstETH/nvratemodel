from os.path import dirname, exists
from os import makedirs
import numpy as np

def ensure_dir(file_path):
    """If a file is to be saved to file_path, this function
    ensures that the folder specified in this path exists or
    is created if it does not. The valid file_path is returned.
    """
    directory = dirname(file_path)
    if not exists(directory):
        makedirs(directory)
    return file_path

def getIndx(value, array):
    """return index in array which is closest to the given value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def sortarrays(sortarray, *sortaccordingly, order='decr'):
    """
    input: any number of np.arrays or lists
    
    output: all arrays are sorted in the same manner and according to values
    in first array.
    return tuple of np.arrays, first element is the sorted sortarray.

    order: 'incr' or 'decr'
    """
    sortarray = np.array(sortarray)
    sortaccordinglyList = (np.array(arr) for arr in sortaccordingly)
    inds = sortarray.argsort()
    if order == 'decr':
        inds = inds[::-1]
    return (sortarray[inds], *(arr[inds] for arr in sortaccordinglyList))

def getmycolor(plotidx, plotidxRange=10):
    """
    Use in color=... command in plt plots.
    """
    plotincr = 1/plotidxRange
    coloridx = plotidx%plotidxRange
    RGBcode = [1.0 - 4*(plotincr*coloridx - 0.5)**2,
               (plotincr*coloridx)**3,
               (1 - plotincr*coloridx)**3]
    return RGBcode

def getmylinestyle(plotidx):
    """
    Use in linestyle=... command in plt plots.
    """
    linestyle_dict = {
            'solid':                'solid',      # Same as (0, ()) or '-'
            'dotted':               'dotted',    # Same as (0, (1, 1)) or '.'
            'dashed':               'dashed',    # Same as '--'
            'dashdot':              'dashdot',  # Same as '-.'
            'loosely dotted':        (0, (1, 3)),
            'densely dotted':        (0, (1, 1)), # same as dotted
            'loosely dashed':        (0, (10, 4)),
            'densely dashed':        (0, (7, 1)),
            'loosely dashdotted':    (0, (7, 3, 1, 3)),
            'dashdotted':            (0, (3, 5, 1, 5)),
            'densely dashdotted':    (0, (3, 1, 1, 1)),
            'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
            'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
            'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)),
            'densely dashdotdotdotted': (0, (3, 1, 1, 1, 1, 1, 1, 1)),
            }
    linestyles = [
            linestyle_dict['loosely dotted'],
            linestyle_dict['loosely dashed'],
            linestyle_dict['loosely dashdotted'],
            linestyle_dict['solid'],
            linestyle_dict['densely dashed'],
            linestyle_dict['densely dashdotted'],
            linestyle_dict['dotted'],
            linestyle_dict['dashed'],
            linestyle_dict['dashdot'],
            linestyle_dict['densely dashdotdotdotted'],
            ]
    return linestyles[plotidx%len(linestyles)]

def printMatrix(M, spacersize = 15, decimals=2):
    if M.ndim == 1:
        print('\n'.join([f'{item:.{decimals}f}' for item in M]))
    elif M.ndim == 2:
        for row in M:
            line = ''
            for item in row:
                element = ''
                if item == 0:
                    element = '0'
                elif np.abs(item)<1e-2:
                    element = '0.'
                elif np.isreal(item):
                    element = f'{np.real(item):.{decimals}f}'
                elif np.real(item) == 0:
                    element = f'{np.imag(item):.{decimals}f}j'
                else:
                    element = f'{item:.{decimals}f}'
                spacer = ''.join([' ' for i in range(spacersize-len(element))])
                line += element+spacer
            print(line)
    else:
        raise NotImplementedError
