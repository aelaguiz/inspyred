def get_pop_size(args):
    grid_size = args['nbh_grid_size']
    return grid_size ** 2


def get_neighbors(pop, i, args):
    return [
        n for i, n in _get_neighbors(pop, i, args)]


def _get_neighbors(pop, i, args):
    grid_size = args['nbh_grid_size']
    nbh_size = args['nbh_size']

    start_pos = (
        i / grid_size,
        i % grid_size)

    nhbrs = []

    for step in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        pos = tuple(start_pos)

        for j in range(nbh_size):
            pos = (pos[0] + step[0], pos[1] + step[1])

            if pos[0] < 0 or pos[0] >= grid_size or\
                    pos[1] < 0 or pos[1] >= grid_size:
                break

            idx = pos[0] * grid_size + pos[1]

            ind = pop[idx]
            nhbrs.append((idx, ind))

    return nhbrs


def replace_into_neighborhood(pop, idx, ind, args):
    nhbrs = _get_neighbors(pop, idx, args)

    # Strip individuals with no calculated fitness
    nhbrs = [(i, n) for i, n in nhbrs if n.fitness is not None]

    nhbrs = sorted(
        nhbrs, key=lambda n: n[1].fitness, reverse=not args['maximize'])

    new_idx = nhbrs[0][0]
    pop[new_idx] = ind

    return new_idx


def log_neighborhood(population, logger, args):
    pos = 0
    grid_size = args['nbh_grid_size']

    for i in range(grid_size):
        row = ""

        for j in range(grid_size):
            cur = population[pos]
            f = cur.fitness

            if f:
                f = round(f, 2)

            row += str(f).ljust(12) + " "

            pos += 1

        logger.debug(row)
