"""A game's positions one by one, in a window or saved as a GIF: what each `watch` shares.

A `watch` experiment builds its figure with `figure`, steps it through the positions with a
`render.Browser`, and hands that to `finish`: with a prefix the frames go to
``{prefix}_game.gif``, and without one (an empty prefix) a window opens.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gridzero.render import Browser


def figure(prefix: str, size: tuple[float, float]) -> Figure:
    """Return a figure for a window (by `pyplot`) or, given a prefix, one for a file alone.

    Parameters
    ----------
    prefix : str
        Where the frames go, or empty for a window.
    size : (float, float)
        Width and height in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return Figure(figsize=size) if prefix else plt.figure(figsize=size)


def finish(browser: Browser, prefix: str, fps: int = 2) -> None:
    """Save the frames as ``{prefix}_game.gif``, or show the window until it is closed.

    Parameters
    ----------
    browser : Browser
        The figure and its frames, kept referenced here while the window is open.
    prefix : str
        Where the frames go, or empty for a window.
    fps : int, default 2
        Frames a second in the GIF.
    """
    if prefix:
        browser.save(f"{prefix}_game.gif", fps)
        print(f"The frames: {prefix}_game.gif.")
    else:
        plt.show()
