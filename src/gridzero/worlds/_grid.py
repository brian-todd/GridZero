"""Where the cells of a 2-D lattice sit on its grid, for planets' maps and terrain."""

import numpy as np
from numpy.typing import NDArray

from gridzero.core import LatticeSpace


def _grid_place(space: LatticeSpace) -> NDArray[np.float64]:
    """Return the cells' centers shifted by half a cell along each axis: shape (n_cells, 2).

    The grid then spans ``basis @ [0, shape)`` exactly, and cells sit at the same fractions of
    it at every resolution.
    """
    return space.centers + space.basis.sum(axis=1) / 2
