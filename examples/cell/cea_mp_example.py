from random import Random
from time import time
import inspyred
from inspyred.ec.cea_parallel_evaluator import\
    cell_evaluator_mp, cell_evaluator_mp_cleanup
import sys
import os



def evaluator(candidates, args):
    problem = args['problem']
    return problem.evaluator(candidates, args)


def main(prng=None, display=False): 
    if prng is None:
        prng = Random()
        prng.seed(time()) 
    
    #import logging
    #logger = logging.getLogger('inspyred.ec')
    #logger.setLevel(logging.DEBUG)
    #h1 = logging.StreamHandler(sys.stdout)
    #file_handler = logging.FileHandler('inspyred.log', mode='w')
    #file_handler.setLevel(logging.DEBUG)
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #file_handler.setFormatter(formatter)
    #logger.addHandler(h1)
    #logger.addHandler(file_handler)

    problem = inspyred.benchmarks.Ackley(2)

    ea = inspyred.ec.cEA(prng)
    ea.terminator = inspyred.ec.terminators.evaluation_termination

    ea.variator = [inspyred.ec.variators.n_point_crossover, inspyred.ec.variators.bit_flip_mutation]
    ea.observer = inspyred.ec.observers.stats_observer

    final_pop = ea.evolve(
        neighborhood=inspyred.ec.neighborhoods.grid_neighborhood,
        generator=problem.generator,

        evaluator=cell_evaluator_mp,

        mp_evaluator=evaluator, 

        problem=problem,

        mp_num_cpus=8,

        nbh_grid_size=10,
        nbh_size=1,

        maximize=problem.maximize,
        bounder=problem.bounder,
        max_evaluations=30000, 
        num_elites=1)
                          
    if display:
        best = max(final_pop)
        print('Best Solution: \n{0}'.format(str(best)))

 
    cell_evaluator_mp_cleanup()

    return ea
            
if __name__ == '__main__':
    main(display=True)
