import random
from functools import lru_cache
from itertools import count
from operator import itemgetter

from player.tetris_genome import TetrisGenome
from tetris.tetris import Actions, Tetris


def play(game: Tetris, brain: TetrisGenome, time_out=None, tick_callback=None) -> Tetris:
    def get_col(x):
        return tuple(map(bool, game.get_col(x)))

    for tick in count():
        game.t = tick
        if game.game_over or (time_out and tick > time_out):
            break

        moves_per_tick = max(1, (10 - game.level))

        for _ in range(moves_per_tick):
            columns = tuple(get_col(x) for x in range(game.w))

            output_activations = brain(inputs=columns)

            action_idx = max(
                enumerate(output_activations),
                key=itemgetter(1)
            )[0]
            action = list(Actions)[action_idx]
            game.action(action)
            if action in [Actions.DROP_ONE, Actions.DROP_TO_BOTTOM]:
                break  # go to next tick
        else:
            game.drop()  # tetromino drops 1 unit automatically at the end of the tick

        if tick_callback:
            tick_callback(game)

    return game
