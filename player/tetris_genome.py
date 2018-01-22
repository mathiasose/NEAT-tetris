import random
from functools import lru_cache
from itertools import count
from operator import itemgetter

from neat.checkpoint import Checkpointer
from neat.config import Config
from neat.genome import DefaultGenome
from neat.nn.feed_forward import FeedForwardNetwork
from neat.population import Population
from neat.reporting import StdOutReporter
from neat.reproduction import DefaultReproduction
from neat.species import DefaultSpeciesSet
from neat.stagnation import DefaultStagnation
from neat.statistics import StatisticsReporter

from tetris.tetris import Actions, O, Tetris


class PoolingGenome(DefaultGenome):
    pass


class DecisionGenome(DefaultGenome):
    pass


class TetrisGenomeConfig(object):
    """
    This is just a dummy
    """
    __params = []

    def __init__(self, params):
        self.output_keys = []


class TetrisGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        return TetrisGenomeConfig(param_dict)

    def __init__(self, key):
        self.pooling_genome = PoolingGenome(key='{}_pooling'.format(key))
        self.decision_genome = DecisionGenome(key='{}_decision'.format(key))

        super().__init__(key)

    def configure_new(self, config):
        self.pooling_genome.configure_new(
            config=POOL_CONFIG.genome_config,
        )
        self.decision_genome.configure_new(
            config=DECISION_CONFIG.genome_config,
        )

    def configure_crossover(self, genome1, genome2, config):
        self.pooling_genome.configure_crossover(
            genome1=genome1.pooling_genome,
            genome2=genome2.pooling_genome,
            config=None,
        )

        self.decision_genome.configure_crossover(
            genome1=genome1.decision_genome,
            genome2=genome2.decision_genome,
            config=None,
        )

    def mutate(self, config):
        self.pooling_genome.mutate(POOL_CONFIG.genome_config)
        self.decision_genome.mutate(DECISION_CONFIG.genome_config)

    def distance(self, other, config):
        pool_distance = self.pooling_genome.distance(
            other=other.pooling_genome,
            config=POOL_CONFIG.genome_config,
        )
        decision_distance = self.decision_genome.distance(
            other=other.decision_genome,
            config=DECISION_CONFIG.genome_config,
        )

        # TODO decide if sum, avg or something else makes the most sense
        return pool_distance + decision_distance

    @property
    def fitness(self):
        assert self.decision_genome.fitness == self.pooling_genome.fitness
        return self.decision_genome.fitness

    @fitness.setter
    def fitness(self, value):
        self.pooling_genome.fitness = self.decision_genome.fitness = value

    def __str__(self):
        return "Pooling genome:\n{}\n\nDecision genome:\n{}".format(
            '\t' + str(self.pooling_genome).replace('\n', '\n\t'),
            '\t' + str(self.decision_genome).replace('\n', '\n\t'),
        )


MAIN_CONFIG = Config(
    TetrisGenome,
    DefaultReproduction,
    DefaultSpeciesSet,
    DefaultStagnation,
    'config',
)

DECISION_CONFIG = Config(
    DecisionGenome,
    DefaultReproduction,
    DefaultSpeciesSet,
    DefaultStagnation,
    'config',
)

POOL_CONFIG = Config(
    PoolingGenome,
    DefaultReproduction,
    DefaultSpeciesSet,
    DefaultStagnation,
    'config',
)

W = DECISION_CONFIG.genome_config.num_inputs
H = POOL_CONFIG.genome_config.num_inputs


def develop_phenotype(genome, config):
    pooling_net = FeedForwardNetwork.create(
        genome=genome.pooling_genome,
        config=POOL_CONFIG,
    )
    decision_net = FeedForwardNetwork.create(
        genome=genome.decision_genome,
        config=DECISION_CONFIG,
    )

    memoized_pooling_activation = lru_cache()(pooling_net.activate)

    def brain(inputs):
        pooled_layer = [
            memoized_pooling_activation(seq)[0] for seq in inputs
        ]

        decision_layer = decision_net.activate(pooled_layer)

        return decision_layer

    return brain
