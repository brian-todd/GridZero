"""The planet's evolution once more, as JAX arrays: for many planets at once on the GPU.

`Evolution` stays the model: its experiment, its tests and its pictures read it.
`EvolutionArrays` gives the same phases as rules of `gridzero.batched`, which read a state's
`gridzero.batched.Fields` and return the intents `core`'s rules return. A
`gridzero.batched.Simulation` steps them for one planet, and ``jax.vmap`` for a batch: other
seeds, or other parameters.

Amounts are float32 here, so a run follows the reference to float32's precision, until a
trait that the reference keeps just above ``minimum`` dies out here, or the other way round.
"""

import jax
import jax.numpy as jnp

from gridzero import batched
from gridzero.batched import Context, Fields, Phase, Source, Spread
from gridzero.core import Adjacency

from . import Evolution, EvolutionState


class EvolutionArrays:
    """The phases of an `Evolution` as rules in arrays.

    Parameters
    ----------
    model : Evolution
        The parameters. Its start is the reference's: ``batched.fields(model.start(space,
        seed))``.

    Attributes
    ----------
    model : Evolution
        The parameters.

    Raises
    ------
    TypeError
        If ``model`` is not an `Evolution`.
    """

    def __init__(self, model: Evolution) -> None:
        if not isinstance(model, Evolution):
            raise TypeError(f"model must be an Evolution, got {type(model).__name__}")
        self.model = model

    def phases(self) -> list[Phase[EvolutionState]]:
        """Return the phases of a step, as `Evolution.phases` gives them.

        Returns
        -------
        list of Phase
            ``warm`` (with ``warming``), ``live``, ``disperse`` and ``die_out``, in that order.
        """
        model = self.model
        heredity = jnp.asarray(model.heredity(), jnp.float32)
        crowding = jnp.asarray(model.crowding(), jnp.float32)
        traits = jnp.asarray(model.traits, jnp.float32)
        life, temperature = EvolutionState.life, EvolutionState.temperature

        @batched.phase(writes=temperature)
        def warm(s: Fields[EvolutionState], ctx: Context) -> batched.Assign:
            """Raise the temperature everywhere by ``warming``, from ``warming_from`` on."""
            rise = jnp.where(s.step >= model.warming_from, model.warming, 0.0)
            return batched.Assign(temperature, s[temperature] + rise)

        @batched.phase(writes=life)
        def live(s: Fields[EvolutionState], ctx: Context) -> Source:
            """Give births by fitness, some into neighboring traits, and deaths by crowding."""
            apart = (traits[:, None] - s[temperature][None, :]) / model.tolerance
            born = model.births * jnp.exp(-0.5 * apart**2) * s[life]
            crowded = crowding @ (s[life] / ctx.measures)
            died = s[life] * (model.deaths + model.births * crowded / model.capacity)
            return Source(life, heredity @ born - died)

        @batched.phase(writes=life, hoods={"near": Adjacency()})
        def disperse(s: Fields[EvolutionState], ctx: Context) -> Spread:
            """Move a share of each cell's individuals to neighboring land."""
            land = s[EvolutionState.height] > 0
            return Spread(life, ctx.hoods["near"], leave=model.dispersal, domain=land)

        @batched.phase(writes=life)
        def die_out(s: Fields[EvolutionState], ctx: Context) -> Source:
            """Remove each trait too rare in a cell to live on."""
            gone = s[life] / ctx.measures < model.minimum
            return Source(life, -s[life] * gone)

        return [*([warm] if model.warming else []), live, disperse, die_out]


def spectrum(fields: Fields[EvolutionState]) -> jax.Array:
    """Return a planet's individuals by trait, as `spectrum` of the model's module does.

    Parameters
    ----------
    fields : Fields
        One planet's fields.

    Returns
    -------
    Array of float32, shape (BINS,)
        Individuals of each trait, summed over the cells.
    """
    return fields[EvolutionState.life].sum(axis=-1)
