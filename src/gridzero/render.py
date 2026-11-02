"""Pictures of spaces: globes and world maps of spherical meshes, and flat maps of lattices.

A `Raster` records which cell each pixel of an image shows. `globe` views a spherical mesh from
far away (an orthographic projection), and `world_map` unrolls it, in the Mollweide
(equal-area) or the equirectangular projection. `flat_map` draws a 2-D lattice as it lies in
the plane, as pixels. Paint a raster with one color per cell, from `colorize` for values or
`mix` for the shares of groups, and show it with matplotlib. `cells` draws a 2-D lattice as
polygons instead, and `lattice_text` as text. `Browser` steps a figure through frames, such as
the rounds of a game, with a slider and the keyboard, or saves them as an animation.

Each pixel shows the cell the space's `locate` finds at the point the pixel looks at: the cell
containing it on a lattice, and the one with the nearest center on a mesh. On a spherical
Voronoi mesh, such as `icosphere`, that is also the cell containing the point, so pictures are
exact up to the pixel size. On other meshes, cells are drawn as the Voronoi cells of their
centers.

Example
-------
>>> import matplotlib.pyplot as plt
>>> from gridzero.core import icosphere
>>> planet = icosphere(16)
>>> fig, ax = plt.subplots()
>>> image = globe(planet, view=(30, 20)).show(ax, colorize(planet.centers[:, 2]))
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, SupportsIndex

import matplotlib
import matplotlib.colors
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, KeyEvent
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.typing import ColorType
from matplotlib.widgets import Slider
from numpy.typing import ArrayLike, NDArray

from gridzero.core import LatticeSpace, MeshSpace
from gridzero.core._arrays import (
    check_cell,
    check_int,
    check_real,
    finite_floats,
    read_only,
)
from gridzero.core.geometry import sphere_points
from gridzero.core.mesh import check_spherical

type Projection = Literal["orthographic", "mollweide", "equirectangular", "flat"]

_AMBIENT = 0.3
"""Brightness of the parts of a globe that the lamp does not reach."""

_BORDER = 0.5
"""Brightness of the pixels on cell borders, when borders are drawn."""


@dataclass(frozen=True, eq=False)
class Raster:
    """Which cell each pixel of an image shows. Build one with `globe`, `world_map` or `flat_map`.

    Parameters
    ----------
    cells : ndarray of int64, shape (height, width)
        The cell each pixel shows, or -1 for a pixel that shows none: off the sphere, or
        beyond the edge of a lattice that does not wrap around.
    points : ndarray of float64, shape (height, width, 3) or (height, width, 2)
        The point each pixel shows, or NaN where it shows no cell: a point of the unit sphere,
        or of the plane on a flat map.
    light : ndarray of float64, shape (height, width)
        How brightly each pixel is lit, in [0, 1]: by a lamp above and to the left of the
        viewer on a globe, 1 everywhere on a map, and 0 where a pixel shows no cell.
    n_cells : int
        Number of cells of the space.
    projection : {"orthographic", "mollweide", "equirectangular", "flat"}
        How the sphere was flattened, or ``"flat"`` for a grid in the plane.
    extent : tuple of 4 floats
        ``(left, right, bottom, top)`` of the image, for ``imshow``. An equirectangular map
        measures it in degrees of longitude and latitude, and a flat map in the coordinates of
        the lattice's centers, so their axes show coordinates.
    """

    cells: NDArray[np.int64]
    points: NDArray[np.float64]
    light: NDArray[np.float64]
    n_cells: int
    projection: Projection
    extent: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name in ("cells", "points", "light"):
            read_only(getattr(self, name))

    def image(
        self, colors: ArrayLike, *, shade: bool = True, borders: bool = False
    ) -> NDArray[np.float64]:
        """Paint each pixel with the color of its cell.

        Parameters
        ----------
        colors : array_like of float, shape (n_cells, 3) or (n_cells, 4)
            RGB or RGBA color of each cell, in [0, 1], for example from `colorize` or `mix`.
        shade : bool, default True
            Multiply colors by `light`, so a globe looks round. Maps are evenly lit.
        borders : bool, default False
            Darken the pixels where one cell meets another, to show the cells.

        Returns
        -------
        ndarray of float64, shape (height, width, 4)
            RGBA image, transparent where a pixel shows no cell.

        Raises
        ------
        TypeError
            If ``colors`` does not hold numbers.
        ValueError
            If ``colors`` has the wrong shape, or holds values outside [0, 1].
        """
        colors = _rgba(colors, self.n_cells)
        on = self.cells >= 0
        image = np.zeros((*self.cells.shape, 4))
        image[on] = colors[self.cells[on]]
        if shade:
            image[..., :3] *= self.light[..., None]
        if borders:
            image[_borders(self.cells), :3] *= _BORDER
        return image

    def show(
        self,
        ax: Axes,
        colors: ArrayLike,
        *,
        shade: bool = True,
        borders: bool = False,
        **kwargs: Any,  # noqa: ANN401 - forwarded to Axes.imshow, which types each one
    ) -> AxesImage:
        """Draw the image on matplotlib axes.

        Globes and Mollweide maps hide the axes. Equirectangular maps keep them, labeled in
        degrees, and flat maps label them x and y. To animate, update the returned image with
        ``image.set_data(raster.image(colors))``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Where to draw.
        colors : array_like of float, shape (n_cells, 3) or (n_cells, 4)
            Color of each cell, as for `image`.
        shade, borders : bool
            As for `image`.
        **kwargs
            Passed to ``ax.imshow``.

        Returns
        -------
        matplotlib.image.AxesImage
        """
        kwargs.setdefault("extent", self.extent)
        kwargs.setdefault("interpolation", "nearest")
        image = ax.imshow(self.image(colors, shade=shade, borders=borders), **kwargs)
        if self.projection == "equirectangular":
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
        elif self.projection == "flat":
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            ax.set_axis_off()
        return image


def globe(
    space: MeshSpace, view: tuple[float, float] = (0.0, 0.0), size: SupportsIndex = 512
) -> Raster:
    """Picture the sphere as seen from far away: an orthographic projection.

    Parameters
    ----------
    space : MeshSpace
        A spherical mesh, such as ``icosphere(32)``.
    view : (float, float), default (0.0, 0.0)
        Longitude and latitude in degrees of the point at the center of the picture, as in
        `core.geometry.sphere_points`. North is up, unless a pole is at the center.
    size : int, default 512
        Width and height in pixels.

    Returns
    -------
    Raster
        With ``extent`` ``(-1, 1, -1, 1)``: the globe is the unit disk.

    Raises
    ------
    TypeError
        If ``space`` is not a `MeshSpace`, a coordinate of ``view`` is not a real number, or
        ``size`` is not an integer.
    ValueError
        If ``space`` is not spherical, the latitude of ``view`` is not in [-90, 90], or
        ``size`` is less than 1.
    """
    space = check_spherical(space)
    lon, lat = _place(view)
    size = check_int(size, "size", minimum=1)
    toward = sphere_points(lon, lat)  # from the center of the sphere to the viewer
    east = np.array([-np.sin(np.radians(lon)), np.cos(np.radians(lon)), 0.0])
    north = np.cross(toward, east)
    across = (np.arange(size) + 0.5) / size * 2 - 1  # pixel centers, from -1 to 1
    x, y = np.meshgrid(across, -across)  # rows run from top to bottom
    depth = 1 - x**2 - y**2
    on = depth > 0
    z = np.sqrt(np.where(on, depth, 0.0))
    points = x[..., None] * east + y[..., None] * north + z[..., None] * toward
    lamp = toward + 0.5 * north - 0.5 * east
    lamp /= np.linalg.norm(lamp)
    light = _AMBIENT + (1 - _AMBIENT) * np.clip(points @ lamp, 0.0, 1.0)
    return _raster(space, points, on, np.where(on, light, 0.0), "orthographic", (-1, 1, -1, 1))


def world_map(
    space: MeshSpace,
    width: SupportsIndex = 1024,
    *,
    projection: Literal["mollweide", "equirectangular"] = "mollweide",
    center: float = 0.0,
) -> Raster:
    """Picture the whole sphere unrolled onto a map, twice as wide as it is high.

    Parameters
    ----------
    space : MeshSpace
        A spherical mesh, such as ``icosphere(32)``.
    width : int, default 1024
        Width in pixels, at least 2. The height is ``width // 2``.
    projection : {"mollweide", "equirectangular"}, default "mollweide"
        Mollweide maps keep areas in proportion, inside an ellipse. Equirectangular maps fill
        the rectangle, with longitude and latitude as x and y, and stretch areas near the
        poles.
    center : float, default 0.0
        Longitude in degrees at the middle of the map.

    Returns
    -------
    Raster
        On an equirectangular map, ``extent`` is ``(center - 180, center + 180, -90, 90)``.

    Raises
    ------
    TypeError
        If ``space`` is not a `MeshSpace`, ``width`` is not an integer, or ``center`` is not a
        real number.
    ValueError
        If ``space`` is not spherical, ``width`` is less than 2, ``projection`` is unknown, or
        ``center`` is not finite.
    """
    space = check_spherical(space)
    width = check_int(width, "width", minimum=2)
    center = check_real(center, "center")
    height = width // 2
    across = (np.arange(width) + 0.5) / width * 2 - 1  # pixel centers, from -1 to 1
    down = 1 - (np.arange(height) + 0.5) / height * 2  # from 1 at the top to -1
    u, v = np.meshgrid(across, down)
    if projection == "mollweide":
        on = u**2 + v**2 < 1  # the ellipse, scaled to the unit disk
        theta = np.arcsin(np.clip(v, -1.0, 1.0))
        lat = np.degrees(np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi))
        lon = center + 180 * u / np.where(on, np.cos(theta), 1.0)
        extent = (-2 * np.sqrt(2), 2 * np.sqrt(2), -np.sqrt(2), np.sqrt(2))
    elif projection == "equirectangular":
        on = np.ones(u.shape, dtype=bool)
        lon, lat = center + 180 * u, 90 * v
        extent = (center - 180, center + 180, -90.0, 90.0)
    else:
        raise ValueError(f"projection must be 'mollweide' or 'equirectangular', got {projection!r}")
    points = sphere_points(lon, lat)
    return _raster(space, points, on, on.astype(np.float64), projection, extent)


def flat_map(space: LatticeSpace, width: SupportsIndex = 1024) -> Raster:
    """Picture a 2-D lattice as it lies in the plane, with x to the right and y up.

    Squares, hexagons and triangles are drawn exactly, up to the pixel size. Along an axis
    that wraps around, the picture spans one period, so it shows each cell once: a hex grid
    that wraps along x, stored as a rhombus, becomes a rectangle. Along an axis that does not
    wrap, it spans the cells' whole extent, and pixels beyond the grid show nothing, such as
    the notches between the triangles at the ends of the rows. A hex grid that
    wraps along y alone has a slanted period, which no rectangle shows exactly: some cells
    along its top and bottom edges lose a sliver (up to 2% of a cell at 8 x 6) or are drawn in
    two pieces.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice of squares, hexagons or triangles.
    width : int, default 1024
        Width in pixels, at least 1. The height follows, so that pixels are about square.
        Give each cell a few pixels to show its shape.

    Returns
    -------
    Raster
        With ``extent`` in the coordinates of ``space.centers``.

    Raises
    ------
    TypeError
        If ``space`` is not a `LatticeSpace`, or ``width`` is not an integer.
    ValueError
        If ``space`` is not 2-D, or ``width`` is less than 1.
    """
    if not isinstance(space, LatticeSpace):
        raise TypeError(f"need a 2-D LatticeSpace, got {type(space).__name__}")
    if space.dim != 2:
        raise ValueError(f"need a 2-D LatticeSpace, got one with {space.dim} axes")
    width = check_int(width, "width", minimum=1)
    left, right, bottom, top = _flat_extent(space)
    height = max(1, round(width * (top - bottom) / (right - left)))
    across = left + (np.arange(width) + 0.5) / width * (right - left)  # pixel centers
    down = top - (np.arange(height) + 0.5) / height * (top - bottom)  # rows from the top
    points = np.stack(np.meshgrid(across, down), axis=-1)
    on = np.ones((height, width), dtype=bool)
    return _raster(space, points, on, on.astype(np.float64), "flat", (left, right, bottom, top))


def cells(
    space: LatticeSpace,
    colors: ArrayLike,
    ax: Axes | None = None,
    *,
    edgecolor: ColorType = (0.0, 0.0, 0.0, 0.12),
    linewidth: float = 0.4,
) -> Axes:
    """Draw a 2-D lattice cell by cell, each as its polygon (`LatticeSpace.corners`).

    Polygons stay sharp at any size, unlike a `flat_map`, so this suits small boards and
    pieces drawn on top of them. The axes get an equal aspect, limits that hold every cell,
    and no frame.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice of squares, hexagons or triangles.
    colors : array_like of float, shape (n_cells, 3) or (n_cells, 4)
        Each cell's color, RGB or RGBA in [0, 1].
    ax : matplotlib.axes.Axes, optional
        Where to draw. By default, on a new figure of its own, made without `pyplot`.
    edgecolor : color, default faint black
        The color of the cells' edges.
    linewidth : float, default 0.4
        The width of the cells' edges, in points, at least 0.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on. Its first new collection holds one polygon per cell, in order of
        cell id.

    Raises
    ------
    ValueError
        If ``space`` is not 2-D, or ``colors`` has the wrong shape or values outside [0, 1].
    """
    corners = space.corners()
    facecolors = _rgba(colors, space.n_cells)
    linewidth = check_real(linewidth, "linewidth", 0.0)
    if ax is None:
        ax = Figure(figsize=(12, 7)).add_subplot()
    polygons = PolyCollection(
        list(corners), facecolors=facecolors, edgecolors=edgecolor, linewidths=linewidth
    )
    ax.add_collection(polygons)  # which asks the axes to fit their limits to it
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


class Browser:
    """Step a figure through its frames: with a slider, the keyboard, or by playing them.

    The figure keeps a slider along its bottom, below 0.06 of its height: leave that free, as
    with ``figure.subplots_adjust(bottom=0.12)``. Keys: right and left step a frame, home and
    end go to the first and the last, and space plays the frames in turn or pauses. In a window,
    keep the browser referenced while it is open: matplotlib holds its widgets only weakly.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to draw on, from ``pyplot.figure()`` to show it in a window.
    frames : int
        How many frames, at least 2.
    draw : callable
        ``draw(frame)`` draws the frame of that number, from 0, on the figure's axes, for
        example by updating images and lines in place. The browser draws frame 0 when made.
    label : str, default "frame"
        The slider's label.
    interval : int, default 500
        Milliseconds between frames while playing, at least 1.

    Attributes
    ----------
    slider : matplotlib.widgets.Slider
        The slider, whose value is the frame shown.
    playing : bool
        Whether the frames are playing.
    timer : matplotlib.backend_bases.TimerBase
        The timer that shows the next frame while playing, every ``interval`` milliseconds.

    Raises
    ------
    TypeError, ValueError
        If ``frames`` or ``interval`` is not an integer of at least 2 or 1.
    """

    def __init__(
        self,
        figure: Figure,
        frames: SupportsIndex,
        draw: Callable[[int], object],
        *,
        label: str = "frame",
        interval: SupportsIndex = 500,
    ) -> None:
        self.figure = figure
        self.n_frames = check_int(frames, "frames", minimum=2)
        self._draw = draw
        self.slider = Slider(
            figure.add_axes((0.2, 0.02, 0.6, 0.03)),
            label,
            0,
            self.n_frames - 1,
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._changed)
        self.playing = False
        self.timer = figure.canvas.new_timer(interval=check_int(interval, "interval", 1))
        self.timer.add_callback(self._tick)
        figure.canvas.mpl_connect("key_press_event", self._key)
        draw(0)

    @property
    def frame(self) -> int:
        """The frame shown."""
        return int(self.slider.val)

    def go(self, frame: SupportsIndex) -> None:
        """Show a frame.

        Parameters
        ----------
        frame : int
            Its number, from 0 to ``n_frames - 1``.

        Raises
        ------
        TypeError, ValueError
            If ``frame`` is not an integer in that range.
        """
        frame = check_int(frame, "frame", minimum=0)
        if frame >= self.n_frames:
            raise ValueError(f"frame must be less than {self.n_frames}, got {frame}")
        self.slider.set_val(frame)

    def redraw(self) -> None:
        """Draw the frame shown again, after what ``draw`` shows has changed."""
        self._changed(self.frame)

    def play(self) -> None:
        """Play the frames in turn from the one shown, or pause them if they are playing.

        Played from the last frame, they start again from the first.
        """
        if self.playing:
            self.playing = False
            self.timer.stop()
            return
        if self.frame == self.n_frames - 1:
            self.go(0)
        self.playing = True
        self.timer.start()

    def save(self, path: str | Path, fps: SupportsIndex = 2) -> None:
        """Save every frame in turn as an animated GIF, leaving the first one shown.

        Parameters
        ----------
        path : str or Path
            Where to save it.
        fps : int, default 2
            Frames a second, at least 1.

        Raises
        ------
        TypeError, ValueError
            If ``fps`` is not an integer of at least 1.
        """
        fps = check_int(fps, "fps", minimum=1)
        frames = range(self.n_frames)
        animation = FuncAnimation(self.figure, self.go, frames, cache_frame_data=False)
        animation.save(path, writer=PillowWriter(fps=fps))
        self.go(0)

    def _changed(self, value: float) -> None:
        """Draw the frame the slider moved to."""
        self._draw(int(value))
        self.figure.canvas.draw_idle()

    def _tick(self) -> None:
        """Show the next frame while playing, and stop at the last."""
        if self.frame < self.n_frames - 1:
            self.go(self.frame + 1)
        if self.frame == self.n_frames - 1:
            self.playing = False
            self.timer.stop()

    def _key(self, event: Event) -> None:
        """Step, jump, or play as the key pressed says."""
        if not isinstance(event, KeyEvent):
            return
        moves = {
            "right": min(self.frame + 1, self.n_frames - 1),
            "left": max(self.frame - 1, 0),
            "home": 0,
            "end": self.n_frames - 1,
        }
        if event.key in moves:
            if moves[event.key] != self.frame:
                self.go(moves[event.key])
        elif event.key == " ":
            self.play()


def lattice_text(
    space: LatticeSpace,
    glyphs: ArrayLike,
    *,
    around: SupportsIndex | None = None,
    radius: SupportsIndex = 8,
) -> str:
    """Draw a 2-D lattice as text, one character per cell, with its coordinates.

    Rows run along axis 0 and are written top down, the last row first, as in pictures. Each
    row of hexagons is shifted right by half a cell (one character) more than the one below,
    so the rhombus looks like one. Under the bottom row, two lines give the tens and the units
    of the coordinate along axis 0 (``q`` on hexes, ``i`` otherwise), or only the units when
    no coordinate drawn reaches 10; each row starts with its coordinate along axis 1.

    Parameters
    ----------
    space : LatticeSpace
        A 2-D lattice.
    glyphs : array_like of str, shape (n_cells,)
        One character per cell.
    around : int, optional
        A cell to draw the lattice around instead of drawing it whole: the cells whose
        coordinates both lie within ``radius`` of its own (a box of lattice coordinates, cut at
        the lattice's edges, also where an axis wraps).
    radius : int, default 8
        How far from ``around`` to draw, at least 0. Unused without ``around``.

    Returns
    -------
    str
        The lines, joined by newlines, without a final one.

    Raises
    ------
    ValueError
        If ``space`` is not 2-D, or ``glyphs`` is not one character for each cell.
    TypeError
        If ``radius`` is not an integer.
    IndexError
        If ``around`` is not a cell of the lattice.
    """
    if space.dim != 2:
        raise ValueError(f"lattice text needs a 2-D lattice, got {space.dim}-D")
    chars = np.asarray(glyphs)
    if chars.shape != (space.n_cells,):
        raise ValueError(f"glyphs must have shape ({space.n_cells},), got {chars.shape}")
    if chars.dtype.kind != "U" or np.any(np.char.str_len(chars) != 1):
        raise ValueError("glyphs must be one character per cell")
    q_len, r_len = space.shape
    q_lo, r_lo = 0, 0
    span = check_int(radius, "radius", 0)
    if around is not None:
        q0, r0 = (int(x) for x in space.unravel(check_cell(around, space.n_cells)))
        q_lo, r_lo = max(q0 - span, 0), max(r0 - span, 0)
        q_len, r_len = min(q0 + span + 1, q_len), min(r0 + span + 1, r_len)
    grid = space.grid_view(chars)  # (Q, R): axis 0 along each row
    hexes = space.tiling == "hex"
    lines = [
        f"{r:>3} " + " " * (r - r_lo if hexes else 0) + " ".join(grid[q_lo:q_len, r])
        for r in range(r_len - 1, r_lo - 1, -1)
    ]
    label = "q" if hexes else "i"
    units = " ".join(str(q % 10) for q in range(q_lo, q_len))
    if q_len > 10:
        lines.append(
            label + "   " + " ".join(str(q // 10) if q >= 10 else " " for q in range(q_lo, q_len))
        )
        label = " "
    lines.append(label + "   " + units)
    return "\n".join(lines)


def colorize(
    values: ArrayLike,
    cmap: str | Colormap = "viridis",
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> NDArray[np.float64]:
    """Return a color for each value, from a matplotlib colormap.

    Parameters
    ----------
    values : array_like of float, shape (n_cells,)
        One value per cell. NaN marks a cell with no value.
    cmap : str or matplotlib.colors.Colormap, default "viridis"
        Colormap, or its name.
    vmin, vmax : float, optional
        Values at the two ends of the colormap. They default to the smallest and largest
        value other than NaN.

    Returns
    -------
    ndarray of float64, shape (n_cells, 4)
        RGBA colors in [0, 1], transparent where a value is NaN.

    Raises
    ------
    TypeError
        If ``values`` does not hold numbers.
    ValueError
        If ``values`` is not 1-D, or ``cmap`` is not a known colormap.
    """
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        raise TypeError(f"values must hold numbers, got dtype {values.dtype}")
    if values.ndim != 1:
        raise ValueError(f"values must have one entry per cell, got shape {values.shape}")
    values = values.astype(np.float64)
    colormap = matplotlib.colormaps.get_cmap(cmap)
    known = ~np.isnan(values)
    if not known.any():
        return np.zeros((len(values), 4))
    low = np.min(values[known]) if vmin is None else check_real(vmin, "vmin")
    high = np.max(values[known]) if vmax is None else check_real(vmax, "vmax")
    colors = np.asarray(colormap(matplotlib.colors.Normalize(low, high)(values)), np.float64)
    colors[~known] = 0.0
    return colors


def mix(weights: ArrayLike, palette: Sequence[ColorType] | ArrayLike) -> NDArray[np.float64]:
    """Blend one color per group by each cell's weights, such as the people of each group.

    Parameters
    ----------
    weights : array_like of float, shape (n_groups, n_cells)
        Non-negative weight of each group in each cell. Only ratios within a cell matter.
    palette : sequence of colors, n_groups of them
        Any colors matplotlib understands, such as ``["tab:orange", "gold"]`` or an array of
        shape (n_groups, 3) or (n_groups, 4).

    Returns
    -------
    ndarray of float64, shape (n_cells, 4)
        The weighted average of the palette in each cell, alpha included. Transparent where
        the weights add up to 0.

    Raises
    ------
    TypeError
        If ``weights`` does not hold numbers.
    ValueError
        If ``weights`` is not 2-D with a row per color, or holds negative, NaN or infinite
        values, or ``palette`` holds something that is not a color.
    """
    palette = matplotlib.colors.to_rgba_array(palette)
    weights = finite_floats(weights, "weights")
    if weights.ndim != 2 or len(weights) != len(palette):
        raise ValueError(
            f"weights must have shape ({len(palette)}, n_cells), one row per color, "
            f"got {weights.shape}"
        )
    if weights.size and weights.min() < 0:
        raise ValueError("weights must not be negative")
    total = weights.sum(axis=0)[:, None]
    colors = np.zeros((weights.shape[1], 4))
    np.divide(weights.T @ palette, total, out=colors, where=total > 0)
    return np.clip(colors, 0.0, 1.0)


def over(top: ArrayLike, bottom: ArrayLike) -> NDArray[np.float64]:
    """Lay one set of colors over another, letting the lower show through where it is clear.

    This is alpha compositing ("over"): a color with alpha ``a`` covers that fraction of the
    color below it.

    Parameters
    ----------
    top, bottom : array_like of float, shape (n_cells, 3) or (n_cells, 4)
        RGB or RGBA colors in [0, 1]. RGB colors are opaque.

    Returns
    -------
    ndarray of float64, shape (n_cells, 4)
        The combined colors, transparent only where both are.

    Raises
    ------
    TypeError
        If either does not hold numbers.
    ValueError
        If their shapes differ in the number of cells or are not colors, or they hold values
        outside [0, 1].
    """
    top = _rgba(top, None)
    bottom = _rgba(bottom, len(top))
    cover = top[:, 3:]
    alpha = cover + bottom[:, 3:] * (1 - cover)
    rgb = np.zeros((len(top), 3))
    blended = top[:, :3] * cover + bottom[:, :3] * bottom[:, 3:] * (1 - cover)
    np.divide(blended, alpha, out=rgb, where=alpha > 0)
    return np.clip(np.column_stack([rgb, alpha]), 0.0, 1.0)


def _place(view: tuple[float, float]) -> tuple[float, float]:
    """Return ``view`` as a longitude and a latitude in degrees, checked."""
    try:
        lon, lat = view
    except TypeError, ValueError:
        raise TypeError(f"view must be a (longitude, latitude) pair, got {view!r}") from None
    return check_real(lon, "the longitude of view"), check_real(
        lat, "the latitude of view", -90, 90
    )


def _raster(
    space: MeshSpace | LatticeSpace,
    points: NDArray[np.float64],
    on: NDArray[np.bool_],
    light: NDArray[np.float64],
    projection: Projection,
    extent: tuple[float, float, float, float],
) -> Raster:
    """Find the cell each pixel shows, among the pixels ``on`` the surface.

    ``points`` is float64 of shape (height, width, dim): where each pixel looks on the
    surface, in the space's coordinates. ``on`` and ``light`` are bool and float64 of shape
    (height, width). Pixels off the surface, or beyond a lattice that does not wrap, show cell
    -1, NaN points and no light.
    """
    cells = np.full(on.shape, -1, dtype=np.int64)
    if on.any():
        cells[on] = space.locate(points[on])
    on = cells >= 0  # a lattice that does not wrap ends before the picture does
    points = np.where(on[..., None], points, np.nan)
    light = np.where(on, light, 0.0)
    left, right, bottom, top = (float(edge) for edge in extent)  # NumPy floats too
    return Raster(cells, points, light, space.n_cells, projection, (left, right, bottom, top))


def _flat_extent(space: LatticeSpace) -> tuple[float, float, float, float]:
    """Return ``(left, right, bottom, top)`` of a flat map of a 2-D lattice.

    Along an axis that wraps, the map spans one period, from half a cell before cell 0. Along
    one that does not, it spans every cell whole.
    """
    (length_x, length_y), (wraps_x, wraps_y) = space.shape, space.periodic
    if space.tiling == "hypercubic":
        return (-0.5, length_x - 0.5, -0.5, length_y - 0.5)
    if space.tiling == "triangle":
        column = np.sqrt(3) / 2  # from one column of triangles to the next: half an edge
        left, right = (
            (-column / 2, (length_x - 0.5) * column) if wraps_x else (-column, length_x * column)
        )
        return (left, right, -0.75, (length_y - 0.5) * 1.5)  # rows are 1.5 high, centered
    rise = np.sqrt(3) / 2  # from one row of hexagons to the next
    corner = 1 / np.sqrt(3)  # from a hexagon's center to its top and bottom corners
    right = length_x - 0.5 if wraps_x else length_x - 0.5 + (length_y - 1) / 2  # a rhombus
    if wraps_y:
        return (-0.5, right, -rise / 2, (length_y - 0.5) * rise)
    return (-0.5, right, -corner, (length_y - 1) * rise + corner)


def _borders(cells: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Mark pixels whose right or lower neighbor shows another cell of the sphere.

    ``cells`` is int64 of shape (height, width), -1 off the sphere. Returns bool of the same
    shape.
    """
    border = np.zeros(cells.shape, dtype=bool)
    for here, there in (
        (np.s_[:, :-1], np.s_[:, 1:]),  # right
        (np.s_[:-1, :], np.s_[1:, :]),  # below
    ):
        a, b = cells[here], cells[there]
        border[here] |= (a != b) & (a >= 0) & (b >= 0)
    return border


def _rgba(colors: ArrayLike, n_cells: int | None) -> NDArray[np.float64]:
    """Return per-cell colors as RGBA, checked: shape (n_cells, 3 or 4), values in [0, 1]."""
    colors = finite_floats(colors, "colors")
    wrong_count = n_cells is not None and len(colors) != n_cells
    if colors.ndim != 2 or colors.shape[1] not in (3, 4) or wrong_count:
        expected = "n_cells" if n_cells is None else n_cells
        raise ValueError(
            f"colors must have shape ({expected}, 3) or ({expected}, 4), got {colors.shape}"
        )
    if colors.size and not (colors.min() >= 0 and colors.max() <= 1):
        raise ValueError("colors must lie in [0, 1]")
    if colors.shape[1] == 3:
        colors = np.column_stack([colors, np.ones(len(colors))])
    return colors
