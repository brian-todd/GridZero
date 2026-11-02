"""GridZero: agents that learn games by self-play, and simulations on the same engine.

Games and simulations run on grids of arbitrary polygonal or polytope cells: flat n-D
lattices, hexagonal and triangular grids, and spherical (planet) meshes.

core
    Spaces, neighborhoods, fields, intents and the update loop.
game
    Games beside simulations: the Game and Player protocols, play, series and scoring.
agents
    Players of any game: AlphaZero, MuZero, imitation, PPO, language models, exact values.
worlds
    Planets, terrain and noise.
render
    Pictures of spaces and games.
"""
