import random


def get_pop_size(args):
    grid_size = args['nbh_grid_size']
    return grid_size ** 2


def get_neighbors(pop, i, args):
    nhbrs = _get_neighbors(pop, i, args)

    return [
        n for i, n in nhbrs]


def _get_neighbors(pop, i, args):
    grid_size = args['nbh_grid_size']
    nbh_size = args['nbh_size']

    start_pos = (
        i / grid_size,
        i % grid_size)

    nhbrs = []

    for step in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        r = start_pos[0]
        c = start_pos[1]

        for j in range(nbh_size):

            r = r + step[0]
            c = c + step[1]

            if r < 0:
                r = grid_size - 1
            elif r >= grid_size:
                r = 0

            if c < 0:
                c = grid_size - 1
            elif c >= grid_size:
                c = 0

            idx = (r * grid_size) + c

            ind = pop[idx]
            nhbrs.append((idx, ind))

    return nhbrs


def get_replacement_dest(pop, idx, ind, logger, args):
    actual_nhbrs = _get_neighbors(pop, idx, args)

    logger.debug("Neighborhood for replacement:\n{0}".format(
        get_neighborhood(
            pop, args, current=idx, neighbors=[
                i for (i, n) in actual_nhbrs])))


    # if maximizing, sort in ascending order 
    # lowest fitness at bottom
    nhbrs = sorted(
        actual_nhbrs, key=lambda n: n[1], reverse=not args['maximize'])

    #logger.debug("Fitnesses in sorted order: {0}".format(
        #[n.fitness for (i, n) in nhbrs]))

    # Only replace if we're an improvement or the same
    #if ind >= nhbrs[0][1]:
    new_idx = nhbrs[0][0]

    return new_idx

    #return None


def get_neighborhood(population, args, current=None, neighbors=[]):
    pos = 0
    grid_size = args['nbh_grid_size']

    ret = ""

    for i in range(grid_size):
        row = ""

        for j in range(grid_size):
            cur = population[pos]
            f = cur.fitness

            if f:
                f = round(f, 2)

            f = str(f)

            if pos == current:
                row += ("*" + f + "*").center(8)
            elif pos in neighbors:
                row += ("|" + f + "|").center(8)
            else:
                row += f.center(8)

            row += " "

            pos += 1

        ret += row + "\n"

    return ret
