"""
    ===================================================
    :mod:`cea` -- Cellular Evoluationary Algorithm Base
    ===================================================
    
    This module provides the base code for cellular evolutionary computations
    
    .. Copyright 2012 Inspired Intelligence Initiative

    .. This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.

    .. This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    .. You should have received a copy of the GNU General Public License
       along with this program.  If not, see <http://www.gnu.org/licenses/>.
       
    .. module:: cea
    .. moduleauthor:: Amir Elaguizy <aelaguiz@gmail.com>
"""

import collections
import copy
import functools

from inspyred.ec import Bounder, EvolutionaryComputation, Individual


def cellular_evaluator(evaluate):
    """Wraps a normal inspyred evaluator and transforms it so that it
    is suitable for use in a cellular algorithm.

    The biggest change is that it expects full individual objects
    rather than just candidate objects, it will unwrap them
    and pass the candidate object through to the underlying evaluator.

    This function should be passed the output of an existing wrapped
    evaluator, such that it accepts multiple candidates at once::
    
        [fitness] = evaluate(candidates, args)
        
    It can be used as a decorator in this way:

        @cell_evaluator
        @evaluator
        def evaluate(candidate, args):
            # Implementation of evaluation
            pass

    It may also be used as an argument to the cea like:

        evaluator=inspyred.ec.cellular_evaluator(problem.evaluator)
    """
    @functools.wraps(evaluate)
    def cell_evaluator(individuals, callback_fn, args, block):
        candidates = [
            ind.candidate for i, ind in individuals]

        for (idx, ind), fitness in zip(individuals, evaluate(candidates, args)):
            ind.fitness = fitness

            callback_fn(idx, ind)

    return cell_evaluator


class cEA(EvolutionaryComputation):
    """Represents a basic cellular evolutionary algorithm.

    Public Attributes:
    - *neighborhood* -- the neighborhood module to use
    
    Protected Attributes:
    
    - *_random* -- the random number generator object
    - *_kwargs* -- the dictionary of keyword arguments initialized
      from the *args* parameter in the *evolve* method
    
    """
    def __init__(self, random):
        EvolutionaryComputation.__init__(self, random)

    def init(
            self, generator, evaluator, pop_size, seeds, maximize,
            bounder, neighborhood, **args):
        self._kwargs = args
        self._kwargs['_ec'] = self
        self._kwargs['maximize'] = maximize

        if bounder is None:
            bounder = Bounder()

        self.num_generation_seed = self._kwargs.setdefault(
            'num_generation_seed', 4)

        self.observe_evaluations = self._kwargs.setdefault(
            'observe_evaluations', 100)

        self.async_evaluator = self._kwargs.setdefault(
            'async_evaluator', False)

        self.max_outstanding_individuals = 100
        self.outstanding_individuals = 0
        self.last_generations = 0
        self.num_evaluations = 0
        self.termination_cause = None
        self.generator = generator
        self.evaluator = evaluator
        self.bounder = bounder
        self.maximize = maximize
        self.population = []
        self.archive = []
        self.neighborhood = neighborhood
        self._terminate = False
        self._eval_queue = collections.defaultdict(list)

    def initial_population(self, seeds):
        if seeds is None:
            seeds = []

        if not isinstance(seeds, collections.Sequence):
            seeds = [seeds]

        initial_cs = copy.copy(seeds)

        num_generated = max(self.pop_size - len(seeds), 0)

        i = 0

        self.logger.debug('generating initial population')

        while i < num_generated:
            cs = self.generator(random=self._random, args=self._kwargs)
            ind = Individual(cs, maximize=self.maximize)
            ind.fitness = None

            initial_cs.append(ind)
            i += 1

        return initial_cs

    
    def evolve(self, generator, evaluator, neighborhood, pop_size=None, seeds=None, maximize=True, bounder=None, **args):
        self.pop_size = neighborhood.get_pop_size(args)

        self.init(
            generator, evaluator, pop_size, seeds, maximize, bounder,
            neighborhood, **args)

        self.population = self.initial_population(seeds)

        self.evaluator(
            callback_fn=self.init_eval_callback, individuals=[
                (i, ind) for i, ind in enumerate(self.population)],
            args=self._kwargs, block=True)

        return self.population

    def init_eval_callback(self, idx, ind):
        self.logger.debug("Initial eval complete for {0} {1} {2}".format(
            idx, ind, ind.fitness))

        self.num_evaluations += 1

        outstanding = [p for p in self.population if p.fitness is None]
        if not outstanding:
            self.logger.debug("Initial population evaluated, starting eval loop")
            self.neighborhood.log_neighborhood(
                self.population, self.logger, self._kwargs)
            self.start_eval_loop()

    def enqueue(self, eval_callback, individuals):
        self._eval_queue[eval_callback] += [(i, ind) for i, ind in enumerate(individuals)]
        self.logger.debug("Enqueued {0} individuals for evaluation - Total {1} by {2}".format(
            len(individuals), len(self._eval_queue[eval_callback]), eval_callback))


    def dispatch(self, block=False):
        """Dispatches all queued evaluations to the evaluator

        If the block flag is set the evaluator is expected to sit in a loop
        hitting `callback_fn` until all individuals have been evaluated.

        Otherwise the evaluator should return immediately after initial
        evaluation and not wait for more individuals to be evaluated.
        """
        if block or (
                self.async_evaluator and self.outstanding_individuals <
                self.max_outstanding_individuals):
            queue = self._eval_queue
            self._eval_queue = collections.defaultdict(list)

            for callback, indivs in queue.iteritems():
                send_indivs = indivs

                num_to_send = max(
                    self.max_outstanding_individuals -
                    self.outstanding_individuals,
                    1)

                self.logger.debug("Attempting to dispatch {0}".format(
                    num_to_send))
                # Only send reasonable chunks out at a time
                if len(send_indivs) > num_to_send:
                    requeue = send_indivs[num_to_send:]
                    send_indivs = send_indivs[:num_to_send]

                    self._eval_queue[callback] = requeue

                self.logger.debug("Dispatching {0} individuals {1} outstanding".format(
                    len(send_indivs), self.outstanding_individuals))

                self.outstanding_individuals += len(send_indivs)

                self.evaluator(
                    callback_fn=callback,
                    individuals=send_indivs,
                    args=self._kwargs,
                    block=block)
        else:
            self.logger.debug("Not dispatching {0} {1} {2}".format(
                block, self.async_evaluator, self.outstanding_individuals))

    def start_eval_loop(self):
        while not self._terminate:
            self.dispatch(block=True)

            self.logger.debug(
                "Eval loop came back around, picking a few more to seed")

            individuals = []
            for idx in self._random.sample(range(self.pop_size), self.num_generation_seed):
                ind = self.population[idx]

                individuals.append(ind)

            self.enqueue(self.eval_callback, individuals)


    def eval_callback(self, idx, ind):
        self.outstanding_individuals -= 1

        self.num_evaluations += 1

        self.logger.debug("Evaluation complete on {0}:{1} eval # {2} outstanding inds # {3}".format(
            idx, ind, self.num_evaluations, self.outstanding_individuals))

        self.population[idx] = ind

        if self.check_term():
            return

        nhbrs = self.neighborhood.get_neighbors(self.population, idx, args=self._kwargs)

        parents = [
            p for p in self.selector(
                random=self._random,
                population=nhbrs, args=self._kwargs) if p.fitness is not None]

        parent_cs = [copy.deepcopy(c.candidate) for c in parents]

        offspring_cs = parent_cs
        
        if isinstance(self.variator, collections.Iterable):
            for op in self.variator:
                self.logger.debug('variation using {0} at generation {1} and evaluation {2}'.format(op.__name__, self.num_generations, self.num_evaluations))
                offspring_cs = op(random=self._random, candidates=offspring_cs, args=self._kwargs)
        else:
            self.logger.debug('variation using {0} at generation {1} and evaluation {2}'.format(self.variator.__name__, self.num_generations, self.num_evaluations))
            offspring_cs = self.variator(random=self._random, candidates=offspring_cs, args=self._kwargs)

        self.logger.debug("Variation produced {0} offspring".format(
            len(offspring_cs)))

        offspring = []
        for cs in offspring_cs:
            ind = Individual(cs, maximize=self.maximize)
            ind.fitness = None
            offspring.append(ind)

        self.enqueue(self.replacement_eval_callback, offspring)
        self.dispatch()

    def replacement_eval_callback(self, idx, ind):
        self.logger.debug("Replacement evaluation complete on {0}:{1}".format(
            idx, ind))

        idx = self.neighborhood.replace_into_neighborhood(
            self.population, idx, ind, self._kwargs)

        self.eval_callback(idx, ind)

    def check_term(self):
        self.num_generations = self.num_evaluations / self.pop_size

        if self.num_generations != self.last_generations or\
                self.num_evaluations % self.observe_evaluations == 0:
            self.neighborhood.log_neighborhood(
                self.population, self.logger, self._kwargs)
            if isinstance(self.observer, collections.Iterable):
                for obs in self.observer:
                    self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(obs.__name__, self.num_generations, self.num_evaluations))
                    obs(population=list(self.population), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
            else:
                self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(self.observer.__name__, self.num_generations, self.num_evaluations))
                self.observer(population=list(self.population), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)

            self.last_generations = self.num_generations

        if self._terminate:
            return True

        if self._should_terminate(list(self.population), self.num_generations, self.num_evaluations):
            self.logger.debug("Terminating")
            self._terminate = True
            return True

        return False
