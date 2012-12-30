import random

print "GOT NEIGHBORHOODS"

def get_pop_size(args):
    print "GETTING POP SIZE LOL"
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
        pos = tuple(start_pos)

        for j in range(nbh_size):
            pos = (pos[0] + step[0], pos[1] + step[1])

            if pos[0] < 0 or pos[0] >= grid_size or\
                    pos[1] < 0 or pos[1] >= grid_size:
                continue

            idx = (pos[0] * grid_size) + pos[1]

            ind = pop[idx]
            nhbrs.append((idx, ind))

    return nhbrs


def replace_into_neighborhood(pop, idx, ind, logger, args):
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
    if ind >= nhbrs[0][1]:
        new_idx = nhbrs[0][0]
        pop[new_idx] = ind

        return new_idx

    return None


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
