import collections
import copy
import functools

from inspyred.ec import Bounder, EvolutionaryComputation, Individual


def cellular_evaluator(evaluate):
    @functools.wraps(evaluate)
    def cell_evaluator(individuals, callback_fn, args):
        candidates = [
            ind.candidate for i, ind in individuals]

        for (idx, ind), fitness in zip(individuals, evaluate(candidates, args)):
            ind.fitness = fitness

            callback_fn(idx, ind)

    return cell_evaluator


class cEA(EvolutionaryComputation):
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
        
        self.num_evaluations = 0
        self.termination_cause = None
        self.generator = generator
        self.evaluator = evaluator
        self.bounder = bounder
        self.maximize = maximize
        self.population = []
        self.archive = []
        self.neighborhood = neighborhood
        self.terminate = False
        self.eval_queue = []

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
                (i, ind) for i, ind in enumerate(self.population)], args=self._kwargs)

        return self.population

    def init_eval_callback(self, idx, ind):
        self.logger.debug("Initial eval complete for {0} {1} {2}".format(
            idx, ind, ind.fitness))

        self.num_evaluations += 1

        if not [p for p in self.population if p.fitness is None]:
            self.logger.debug("Initial population evaluated, starting eval loop")
            self.neighborhood.log_neighborhood(
                self.population, self.logger, self._kwargs)
            self.start_eval_loop()

    def start_eval_loop(self):
        while not self.terminate:
            while(self.eval_queue):
                eqi = self.eval_queue.pop(0)

                self.evaluator(
                    callback_fn=eqi[0],
                    individuals=eqi[1],
                    args=self._kwargs)

            self.logger.debug(
                "Eval loop came back around, picking a few more to seed")
            individuals = []
            for idx in self._random.sample(range(self.pop_size), 4):
                ind = self.population[idx]

                individuals.append(ind)

            self.eval_queue.append(
                (
                    self.eval_callback,
                    [(i, ind) for i, ind in enumerate(individuals)]
                )
            )


    def eval_callback(self, idx, ind):
        self.logger.debug("Evaluation complete on {0}:{1}".format(
            idx, ind))

        self.num_evaluations += 1

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

        offspring = []
        for cs in offspring_cs:
            ind = Individual(cs, maximize=self.maximize)
            ind.fitness = None
            offspring.append(ind)

        self.eval_queue.append(
            (
                self.replacement_eval_callback,
                [(idx, ind) for ind in offspring]
            )
        )

    def replacement_eval_callback(self, idx, ind):
        self.logger.debug("Replacement evaluation complete on {0}:{1}".format(
            idx, ind))

        idx = self.neighborhood.replace_into_neighborhood(
            self.population, idx, ind, self._kwargs)

        self.eval_callback(idx, ind)

    def check_term(self):
        if self.terminate:
            return True

        self.num_generations = self.num_evaluations / self.pop_size

        if 0 == self.num_evaluations % self.pop_size:
            self.neighborhood.log_neighborhood(
                self.population, self.logger, self._kwargs)
            if isinstance(self.observer, collections.Iterable):
                for obs in self.observer:
                    self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(obs.__name__, self.num_generations, self.num_evaluations))
                    obs(population=list(self.population), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
            else:
                self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(self.observer.__name__, self.num_generations, self.num_evaluations))
                self.observer(population=list(self.population), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)

        if self._should_terminate(list(self.population), self.num_generations, self.num_evaluations):
            self.logger.debug("Terminating")
            self.terminate = True
            return True

        return False


            # cand = evaluation generator

            # get neighborhood

            # select candidates from neighborhood

            # reproduce via variation

            # replace new candidates into neighborhood
        
    #def evolve(self, generator, evaluator, pop_size=100, seeds=None, maximize=True, bounder=None, **args):
        #self._kwargs = args
        #self._kwargs['_ec'] = self
        
        #if seeds is None:
            #seeds = []
        #if bounder is None:
            #bounder = Bounder()
        
        #self.termination_cause = None
        #self.generator = generator
        #self.evaluator = evaluator
        #self.bounder = bounder
        #self.maximize = maximize
        #self.population = []
        #self.archive = []
        
        ## Create the initial population.
        #if not isinstance(seeds, collections.Sequence):
            #seeds = [seeds]
        #initial_cs = copy.copy(seeds)
        #num_generated = max(pop_size - len(seeds), 0)
        #i = 0
        #self.logger.debug('generating initial population')
        #while i < num_generated:
            #cs = generator(random=self._random, args=self._kwargs)
            #initial_cs.append(cs)
            #i += 1
        #self.logger.debug('evaluating initial population')
        #initial_fit = evaluator(candidates=initial_cs, args=self._kwargs)
        
        #for cs, fit in zip(initial_cs, initial_fit):
            #if fit is not None:
                #ind = Individual(cs, maximize=maximize)
                #ind.fitness = fit
                #self.population.append(ind)
            #else:
                #self.logger.warning('excluding candidate {0} because fitness received as None'.format(cs))
        #self.logger.debug('population size is now {0}'.format(len(self.population)))
        
        #self.num_evaluations = len(initial_fit)
        #self.num_generations = 0
        
        #self.logger.debug('archiving initial population')
        #self.archive = self.archiver(random=self._random, population=list(self.population), archive=list(self.archive), args=self._kwargs)
        #self.logger.debug('archive size is now {0}'.format(len(self.archive)))
        #self.logger.debug('population size is now {0}'.format(len(self.population)))
                
        #if isinstance(self.observer, collections.Iterable):
            #for obs in self.observer:
                #self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(obs.__name__, self.num_generations, self.num_evaluations))
                #obs(population=list(self.population), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
        #else:
            #self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(self.observer.__name__, self.num_generations, self.num_evaluations))
            #self.observer(population=list(self.population), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)

        #while not self._should_terminate(list(self.population), self.num_generations, self.num_evaluations):
            ## Select individuals.
            #self.logger.debug('selection using {0} at generation {1} and evaluation {2}'.format(self.selector.__name__, self.num_generations, self.num_evaluations))
            #parents = self.selector(random=self._random, population=list(self.population), args=self._kwargs)
            #self.logger.debug('selected {0} candidates'.format(len(parents)))
            #parent_cs = [copy.deepcopy(i.candidate) for i in parents]
            #offspring_cs = parent_cs
            
            #if isinstance(self.variator, collections.Iterable):
                #for op in self.variator:
                    #self.logger.debug('variation using {0} at generation {1} and evaluation {2}'.format(op.__name__, self.num_generations, self.num_evaluations))
                    #offspring_cs = op(random=self._random, candidates=offspring_cs, args=self._kwargs)
            #else:
                #self.logger.debug('variation using {0} at generation {1} and evaluation {2}'.format(self.variator.__name__, self.num_generations, self.num_evaluations))
                #offspring_cs = self.variator(random=self._random, candidates=offspring_cs, args=self._kwargs)
            #self.logger.debug('created {0} offspring'.format(len(offspring_cs)))
            
            ## Evaluate offspring.
            #self.logger.debug('evaluation using {0} at generation {1} and evaluation {2}'.format(evaluator.__name__, self.num_generations, self.num_evaluations))
            #offspring_fit = evaluator(candidates=offspring_cs, args=self._kwargs)
            #offspring = []
            #for cs, fit in zip(offspring_cs, offspring_fit):
                #if fit is not None:
                    #off = Individual(cs, maximize=maximize)
                    #off.fitness = fit
                    #offspring.append(off)
                #else:
                    #self.logger.warning('excluding candidate {0} because fitness received as None'.format(cs))
            #self.num_evaluations += len(offspring_fit)        

            ## Replace individuals.
            #self.logger.debug('replacement using {0} at generation {1} and evaluation {2}'.format(self.replacer.__name__, self.num_generations, self.num_evaluations))
            #self.population = self.replacer(random=self._random, population=self.population, parents=parents, offspring=offspring, args=self._kwargs)
            #self.logger.debug('population size is now {0}'.format(len(self.population)))
            
            ## Migrate individuals.
            #self.logger.debug('migration using {0} at generation {1} and evaluation {2}'.format(self.migrator.__name__, self.num_generations, self.num_evaluations))
            #self.population = self.migrator(random=self._random, population=self.population, args=self._kwargs)
            #self.logger.debug('population size is now {0}'.format(len(self.population)))
            
            ## Archive individuals.
            #self.logger.debug('archival using {0} at generation {1} and evaluation {2}'.format(self.archiver.__name__, self.num_generations, self.num_evaluations))
            #self.archive = self.archiver(random=self._random, archive=self.archive, population=list(self.population), args=self._kwargs)
            #self.logger.debug('archive size is now {0}'.format(len(self.archive)))
            #self.logger.debug('population size is now {0}'.format(len(self.population)))
            
            #self.num_generations += 1
            #if isinstance(self.observer, collections.Iterable):
                #for obs in self.observer:
                    #self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(obs.__name__, self.num_generations, self.num_evaluations))
                    #obs(population=list(self.population), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
            #else:
                #self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(self.observer.__name__, self.num_generations, self.num_evaluations))
                #self.observer(population=list(self.population), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
        #return self.population
