import multiprocessing
import random
from functools import lru_cache
from itertools import count
from operator import itemgetter

from neat.checkpoint import Checkpointer
from neat.config import Config
from neat.genome import DefaultGenome
from neat.nn.feed_forward import FeedForwardNetwork
from neat.parallel import ParallelEvaluator
from neat.population import Population
from neat.reporting import StdOutReporter
from neat.reproduction import DefaultReproduction
from neat.species import DefaultSpeciesSet
from neat.stagnation import DefaultStagnation
from neat.statistics import StatisticsReporter

from player.play import play
from player.tetris_genome import MAIN_CONFIG, H, W, develop_phenotype
from tetris.tetris import I, O, Tetris
from math import exp
CPU_COUNT = multiprocessing.cpu_count()


def fitness_f(multi_genome, config):
    brain = develop_phenotype(multi_genome, config)

    easy_tetris = Tetris(w=W, h=H, tetrominoes=[I])

    easy_game = play(game=easy_tetris, brain=brain, time_out=100)

    if not easy_game.lines:
        return easy_game.score / 100

    random.seed(123456)
    proper_tetris = Tetris(w=W, h=H)
    time_out = config.fitness_threshold

    proper_game = play(game=proper_tetris, brain=brain, time_out=time_out)

    return proper_game.score


evaluator = ParallelEvaluator(
    num_workers=CPU_COUNT,
    eval_function=fitness_f,
)

if __name__ == '__main__':
    print('CPU_COUNT:', CPU_COUNT)

    p = Population(MAIN_CONFIG)

    p.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(Checkpointer(50))

    try:
        winner = p.run(evaluator.evaluate, 1000)
    except KeyboardInterrupt:
        pass

    stats.save()

    print(stats.best_genome())
