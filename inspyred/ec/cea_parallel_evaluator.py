import functools
import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

from multiprocessing import TimeoutError

__async_pool = None
__async_results = []


def cell_evaluator_mp(individuals, callback_fn, args):
    """
    Evaluator function will asynchronously dispatch any new individuals
    for evaluation and block while there are outstanding evaluations.

    This functions by maintaining a single global process pool which is
    spun up on demand and then maintained until `cell_evaluator_mp_cleanup`
    is called.

    This allows asynchronous operations to take place because the callback
    into the actual algorithm will allow new asynchronous operations to
    be enqueued in a recursive manner.
    """
    global __async_pool
    global __async_results

    if not __async_pool:
        init_pool(args)

    evaluator = get_evaluator(args)
    pickled_args = get_args(args)

    timeout_val = args.setdefault('mp_timeout_fitness', 0)

    logger = args['_ec'].logger

    for idx, ind in individuals:
        res = __async_pool.apply_async(
            evaluator, ([ind.candidate], pickled_args))

        __async_results.append((idx, ind, res))

    __async_results = list(
        dispatch_results(callback_fn, __async_results, timeout_val))

    while __async_results:
        time.sleep(0.1)

        __async_results = list(
            dispatch_results(callback_fn, __async_results, timeout_val))


def dispatch_results(callback_fn, async_results, timeout_val):
    """Calls the evaluation callback for any asynchronous evaluations which have
    finished processing and returned results.
    """
    for idx, ind, res in list(async_results):
        if res.ready():
            try:
                ret = res.get(0)[0]
                ind.fitness = ret
            except TimeoutError as e:
                ind.fitness = timeout_val
            callback_fn(idx, ind)
        else:
            yield (idx, ind, res)



def get_evaluator(args):
    logger = args['_ec'].logger

    try:
        return args['mp_evaluator']
    except KeyError:
        logger.error('parallel_evaluation_mp requires \'mp_evaluator\' be defined in the keyword arguments list')
        raise 


def get_args(args):
    logger = args['_ec'].logger

    pickled_args = {}
    for key in args:
        try:
            pickle.dumps(args[key])
            pickled_args[key] = args[key]
        except (TypeError, pickle.PickleError, pickle.PicklingError):
            logger.debug('unable to pickle args parameter {0} in parallel_evaluation_mp'.format(key))
            pass

    return pickled_args


def set_pool(pool):
    """Provides a way for injection of a custom process pool
    """
    global __async_pool

    __async_pool = pool


def init_pool(args):
    """Creates the process pool which will be the single global process pool
    used for asynchronous dispatches.
    """
    global __async_pool

    import time
    import multiprocessing
    logger = args['_ec'].logger

    try:
        nprocs = args['mp_nprocs']
    except KeyError:
        nprocs = multiprocessing.cpu_count()

    try:
        __async_pool = multiprocessing.Pool(processes=nprocs)
    except (OSError, RuntimeError) as e:
        logger.error('failed parallel_evaluation_mp: {0}'.format(str(e)))
        raise


def cell_evaluator_mp_cleanup():
    """Cleans up the existing process pool
    """
    global __async_pool

    __async_pool.close()
    __async_pool.join()
