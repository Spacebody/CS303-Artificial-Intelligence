
__author__ = "Yilin Zheng"
__copyright__ = "Copyright (C) 2017 Yilin Zheng"

__revision__ = "$Id$"
__version__ = "1.0.0"

import numpy as np
import os
import re
import copy
import random
import sys
import getopt
import time

# status
CONNECTED = 1
INFTY = sys.maxint
NINFTY = -sys.maxint
UNDEFINED = -1
POPULATION_SIZE = 5000
ALPAH = 0.5
SEED = 0

# parameters should be initialized
NAME = ""
VERTICES = 0
DEPOT = 0
REQUIRED_EDGES = 0
NON_REQUIRED_EDGES = 0
VEHICLES = 0
CAPACITY = 0
TOTAL_COST_OF_REQUIRED_EDGES = 0

# graph information
ARC = set()
AD_MATRIX = []  # adjacency matrix for cost
AD_ARRAY = {}  # adjacency array
COST = {}  # adjacency array for edge cost
DEMAND = {}  # adjacency array for required edge demand
REARC = []  # required edges
NODE = set()  # vertices
RATIO = {}  # ratio
SHORTEST_DIS = []  # matrix for shortest distance
REVERTICES = []


# -------------------------------------------------------
# Read file
# -------------------------------------------------------
def read_file(_file_name):
    """
    :param _file_name: the name of the file being read
    :return: graph, which is a |E|x4 dimensions matrix, with [head, tail, cost, demand] stored each row
    """
    with open(_file_name) as f:
        global NAME
        NAME = re.findall(": (.+?)\n", f.readline())[0]
        global VERTICES
        VERTICES = int(re.findall(": (.+?)\n", f.readline())[0])
        global DEPOT
        DEPOT = int(re.findall(": (.+?)\n", f.readline())[0])
        global REQUIRED_EDGES
        REQUIRED_EDGES = int(re.findall(": (.+?)\n", f.readline())[0])
        global NON_REQUIRED_EDGES
        NON_REQUIRED_EDGES = int(re.findall(": (.+?)\n", f.readline())[0])
        global VEHICLES
        VEHICLES = int(re.findall(": (.+?)\n", f.readline())[0])
        global CAPACITY
        CAPACITY = int(re.findall(": (.+?)\n", f.readline())[0])
        global TOTAL_COST_OF_REQUIRED_EDGES
        TOTAL_COST_OF_REQUIRED_EDGES = int(re.findall(": (.+?)\n", f.readline())[0])
        # print "NAME: %s\nVERTICES: %s\nDEPOT: %s\nREQUIRED EDGES: %s\nNON-REQUIRED EDGES: %s\nVEHICLES: %s\n" \
        #       "CAPACITY: %s\nTOTAL COST OF REQUIRED EDGES: %s" % (NAME, VERTICES, DEPOT, REQUIRED_EDGES,
        #                                                           NON_REQUIRED_EDGES, VEHICLES, CAPACITY,
        #                                                           TOTAL_COST_OF_REQUIRED_EDGES)
        global ARC
        global REARC
        global AD_MATRIX
        global AD_ARRAY
        global COST
        global DEMAND
        global NODE
        global RATIO
        global SHORTEST_DIS
        AD_MATRIX = np.zeros((VERTICES, VERTICES), dtype=int)
        SHORTEST_DIS = np.full((VERTICES, VERTICES), fill_value=INFTY, dtype=int)
        np.fill_diagonal(SHORTEST_DIS, 0)
        graph = set()
        lines = f.readlines()  # drop 9th line which contains no useful data
        for line in lines[1:-1]:
            line = line.strip()
            line = line.split()
            head = int(line[0])
            tail = int(line[1])
            cost = int(line[2])
            demand = int(line[3])
            ARC.add((head, tail))
            ARC.add((tail, head))
            # construct REARC
            if demand:
                REARC.append((head, tail))
                REARC.append((tail, head))
                if head not in REVERTICES:
                    REVERTICES.append(head)
                elif tail not in REVERTICES:
                    REVERTICES.append(tail)
                # construct DEMAND
                if head not in DEMAND:
                    DEMAND[head] = {}
                    DEMAND[head][tail] = demand
                elif tail not in DEMAND[head]:
                    DEMAND[head][tail] = demand
                if tail not in DEMAND:
                    DEMAND[tail] = {}
                    DEMAND[tail][head] = demand
                elif head not in DEMAND[tail]:
                    DEMAND[tail][head] = demand
            # construct AD_MATRIX
            if not AD_MATRIX[head - 1, tail - 1]:
                AD_MATRIX[head - 1, tail - 1] = AD_MATRIX[tail - 1, head - 1] = CONNECTED
            # construct AD_ARRAY
            if head not in AD_ARRAY:
                AD_ARRAY[head] = set()
                AD_ARRAY[head].add(tail)
            else:
                AD_ARRAY[head].add(tail)
            if tail not in AD_ARRAY:
                AD_ARRAY[tail] = set()
                AD_ARRAY[tail].add(head)
            else:
                AD_ARRAY[tail].add(head)
            # construct COST
            if head not in COST:
                COST[head] = {}
                COST[head][tail] = cost
            elif tail not in COST[head]:
                COST[head][tail] = cost
            if tail not in COST:
                COST[tail] = {}
                COST[tail][head] = cost
            elif head not in COST[tail]:
                COST[tail][head] = cost
            graph.add((head, tail, cost, demand))
            NODE = range(1, VERTICES + 1)
        random.shuffle(REARC)
    return graph


# -------------------------------------------------------
# Population initialization
# -------------------------------------------------------
def population_init():
    """
    Population initialization
    :return: a population
    """
    entities = {}
    loads = {}
    costs = {}
    for i in range(1, POPULATION_SIZE+1):
        path, cost, load = RPSH()
        paths = {}
        for j in range(1, len(path)+1):
            paths[j] = path[j-1]
        entities[i] = paths
        loads[i] = load
        costs[i] = cost
    populations = {1: entities, 2: costs, 3: loads}
    return populations


# -------------------------------------------------------
# Random Path Scanning Heuristic
# -------------------------------------------------------
def RPSH():
    """
    Initialize the solution by random path scanning heuristic
    :return: initial solutions which will be further improved
    """
    paths = []  # set of paths
    rearc = copy.copy(REARC)  # Required arc, free
    vehicles = -1  # vehicle number
    loads = {}  # loads of every vehicle
    costs = {}  # costs of every path
    while True:
        src = DEPOT
        vehicles += 1  # k = k + 1
        loads[vehicles] = 0
        costs[vehicles] = 0
        path = []
        while True:
            cost_add = INFTY
            edge_add = ()
            for edge in rearc:
                if loads[vehicles] + DEMAND[edge[0]][edge[1]] <= CAPACITY:
                    d_se = Dijkstra(src, edge[0])
                    if d_se < cost_add:
                        cost_add = d_se
                        edge_add = edge
                    elif d_se == cost_add and better(edge, edge_add, DEPOT, loads[vehicles]):
                        edge_add = edge
            if edge_add:
                path.append(edge_add)
                rearc.remove(edge_add)
                rearc.remove(inverseArc(edge_add))
                loads[vehicles] += DEMAND[edge_add[0]][edge_add[1]]
                costs[vehicles] += COST[edge_add[0]][edge_add[1]] + cost_add
                src = edge_add[1]
            if not rearc or cost_add == INFTY:  # repeat...until free is empty pr d_bar is infinity
                break
        costs[vehicles] += Dijkstra(src, DEPOT)
        paths.append(path)
        if not rearc:  # repeat...until free is empty
            break
    return paths, sum(costs.values()), sum(loads.values())


# -------------------------------------------------------
# randomly apply five rules
# -------------------------------------------------------
def better(tmp_edge, edge, src, load, rule=None):
    """
    Apply PSH rules
    :param tmp_edge: tmp edge we now estimating
    :param edge: edge we store now
    :param src: the source
    :param load: load on the vehicle
    :param rule: rule number, in range 1~5, selected randomly
    :return:
    """
    if rule is None:
        # seed = datetime.datetime.now()
        rule = random.randint(1, 5)
    # apply rule 1~5
    if not edge:
        return True
    else:
        if rule == 1:  # maximize c_ij/r_ij
            return check_ratio(tmp_edge, edge, isMax=True)  # if tmp_edge better than edge return True else False
        elif rule == 2:  # minimize c_ij/r_ij
            return check_ratio(tmp_edge, edge, isMax=False)  # if tmp_edge better than edge, True, while False
        else:
            tmp_edge_cost = Dijkstra(tmp_edge[1], src)
            edge_cost = Dijkstra(edge[1], src)
            if rule == 3:  # maximize return cost
                return tmp_edge_cost > edge_cost
            if rule == 4:  # minimize return cost
                return tmp_edge_cost < edge_cost
            if rule == 5:
                if load > CAPACITY / 2:  # less than half full capacity, apply rule 3
                    return tmp_edge_cost > edge_cost
                else:  # else apply rule 4
                    return tmp_edge_cost < edge_cost


# -------------------------------------------------------
# get inverse arc
# -------------------------------------------------------
def inverseArc(arc):
    """
    :paraam arc: edge
    """
    return arc[::-1]


# -------------------------------------------------------
# get minimize path
# -------------------------------------------------------
def Dijkstra(src, dest):
    """
    Use Dijkstra algorithm to get the shortest path and the cost
    :param src: source of the path
    :param dest: destination of the path
    :return: a shortest path between source and destination
    """
    if SHORTEST_DIS[src-1][dest-1] != INFTY:
        return SHORTEST_DIS[src-1][dest-1]
    depot = src
    costs = {}
    # preds = {src: None}
    unvisited = {}
    visited = [src]
    V = copy.copy(NODE)
    for v in V:
        costs[v] = INFTY
        # preds[v] = None
        unvisited[v] = INFTY
    costs[src] = 0
    unvisited.pop(src)
    for neighbor in AD_ARRAY[src]:
        costs[neighbor] = COST[src][neighbor]
        # preds[neighbor] = src
        unvisited[neighbor] = COST[src][neighbor]
    while unvisited:
        src = min(unvisited, key=unvisited.get)
        visited.append(src)
        unvisited.pop(src)
        for neighbor in AD_ARRAY[src]:
            if neighbor not in visited:
                alt_cost = costs[src] + COST[src][neighbor]
                if alt_cost < costs[neighbor]:
                    costs[neighbor] = alt_cost
                    # preds[neighbor] = src
                    unvisited[neighbor] = alt_cost
    for node in costs:
        SHORTEST_DIS[depot-1][node-1] = costs[node]
    return costs[dest]


# -------------------------------------------------------
# get minimized or maximized ratio of cost over demand
# -------------------------------------------------------
def check_ratio(tmp_edge, edge, isMax=False):
    """
    Return the specific kind of ratio of cost over remaining demand
    :param tmp_edge: the tmp edge we now estimating
    :param edge: the edge we store
    :param isMax: bool flag to choose whether to find max or min
    :type isMax: bool value to determine whether find max or min ratio
    :return: the bool value True or False
    """
    tmp_edge_ratio = COST[tmp_edge[0]][tmp_edge[1]] / DEMAND[tmp_edge[0]][tmp_edge[1]]
    edge_ratio = COST[edge[0]][edge[1]] / DEMAND[edge[0]][edge[1]]
    if isMax:
        return tmp_edge_ratio > edge_ratio  # if tmp_edge has larger ratio, then return True else False
    else:
        return tmp_edge_ratio < edge_ratio  # if tmp_edge has smaller ratio, then return True else False


# -------------------------------------------------------
# calculate the total cost of the entity
# -------------------------------------------------------
def cal_total_cost(entities):
    """
    Calculate the total cost of given entity
    :param entities:
    :return:
    """
    cost = 0
    for i in entities.keys():
        cost += cal_cost(entities[i])
    return cost


# -------------------------------------------------------
# Calculate the demand for a given path
# -------------------------------------------------------
def cal_demand(path):
    """
    Calculate the demand of a given path
    :param path: a given path
    :return: the total demand of the cost
    """
    demand = 0
    for edge in path:
        demand += DEMAND[edge[0]][edge[1]]
    return demand


# -------------------------------------------------------
# Calculate the cost for a given path
# -------------------------------------------------------
def cal_cost(path):
    """
    Calculate the cost for a given path
    :type path: list
    :param path: a given path
    :return: cost of the path
    """
    costs = 0
    if path:
        for edge in path:
            costs += COST[edge[0]][edge[1]]
        for edge_index in range(len(path)-1):
            costs += Dijkstra(path[edge_index][1], path[edge_index+1][0])
        costs += Dijkstra(DEPOT, path[0][0]) + Dijkstra(path[-1][1], DEPOT)
    return costs


# -------------------------------------------------------
# Move operator: Single inversion(SI)
# -------------------------------------------------------
def SI(entity, cost):
    """
    Perform single insertion
    :param entity:
    :return:
    """
    new_cost = cost
    while new_cost >= cost:
        positions = {}
        candidates = {}
        for i in entity.keys():
            positions[i] = []
            candidates[i] = []
            if entity[i][0][0] != DEPOT:
                positions[i] += [0]
            if entity[i][-1][1] != DEPOT:
                positions[i] += [len(entity[i])-1]
            if entity[i][0][0] != DEPOT and entity[i][0][1] != entity[i][1][0]:
                candidates[i] += [0]
            for j in range(1, len(entity[i])):
                if entity[i][j - 1][1] != entity[i][j][0]:
                    positions[i] += [j]
                if j < len(entity[i]) - 1 and entity[i][j][0] != entity[i][j - 1][1] and entity[i][j][1] != \
                        entity[i][j + 1][0]:
                    candidates[i] += [j]
                elif j == len(entity[i]) - 1 and entity[i][j - 1][1] != entity[i][j][0] and entity[i][j][1] != DEPOT:
                    candidates[i] += [j]
        while True:
            # random.seed(SEED)
            choice_pos_route = random.randint(1, len(positions))
            if choice_pos_route:
                break
        choice_pos_pos = random.randint(0, len(positions[choice_pos_route]))
        while True:
            # random.seed(SEED)
            choice_can_route = random.randint(1, len(candidates))
            if candidates[choice_can_route]:
                break
        choice_can_pos = random.randint(0, len(candidates[choice_can_route]) - 1)
        edge = entity[choice_can_route][choice_can_pos]
        # random.seed(SEED)
        if cal_demand(entity[choice_pos_route]) + DEMAND[edge[0]][edge[1]] <= CAPACITY:
            test_entity_1 = copy.deepcopy(entity)
            test_entity_1[choice_pos_route].insert(choice_pos_pos, edge)
            test_entity_1[choice_can_route].remove(edge)
            test_entity_2 = copy.deepcopy(entity)
            test_entity_2[choice_pos_route].insert(choice_pos_pos, edge)
            test_entity_2[choice_can_route].remove(edge)
            if cal_total_cost(test_entity_1) <= cal_total_cost(test_entity_2):
                if cal_total_cost(test_entity_1) < cost:
                    entity[choice_pos_route].insert(choice_pos_pos, edge)
                    entity[choice_can_route].remove(edge)
                    break
            else:
                if cal_total_cost(test_entity_2) < cost:
                    entity[choice_pos_route].insert(choice_pos_pos, inverseArc(edge))
                    entity[choice_can_route].remove(edge)
                    break
    return entity


# -------------------------------------------------------
# Move operator: Double inversion(DI)
# -------------------------------------------------------
def DI(entity):
    """
    Double insertion
    :param entity:
    :return:
    """
    # count = 0
    while True:
        positions = {}
        candidates = {}
        for i in entity.keys():
            positions[i] = []
            candidates[i] = []
            if entity[i][0][0] != DEPOT:
                positions[i] += [0]
            if entity[i][-1][1] != DEPOT:
                positions[i] += [len(entity[i]) - 1]
            if entity[i][0][0] != DEPOT and entity[i][0][1] == entity[i][1][0] and entity[i][1][1] != entity[i][2][0]:
                candidates[i].append([0, 1])
            for j in range(1, len(entity[i])):
                if entity[i][j - 1][1] != entity[i][j][0]:
                    positions[i] += [j]
                if 1 < j < len(entity[i]) - 1 and entity[i][j - 1][0] != entity[i][j - 2][1] and entity[i][j - 1][1] == \
                        entity[i][j][0] and entity[i][j][1] != entity[i][j + 1][0]:
                    candidates[i].append([j - 1, j])
                elif j == len(entity[i]) - 1 and entity[i][j - 1][0] != entity[i][j - 2][1] and entity[i][j - 1][1] == \
                        entity[i][j][0] and entity[i][j][1] != DEPOT:
                    candidates[i].append([j - 1, j])
        # random.seed(SEED)
        while True:
            choice_pos_route = random.randint(1, len(positions))
            if choice_pos_route:
                break
        choice_pos_pos = random.randint(0, len(positions[choice_pos_route]))
        while True:
            # random.seed(SEED)
            choice_can_route = random.randint(1, len(candidates))
            if candidates[choice_can_route]:
                break
        choice_can_pos = random.randint(0, len(candidates[choice_can_route]) - 1)
        choice_can_pos = candidates[choice_can_route][choice_can_pos]
        edges = [entity[choice_can_route][choice_can_pos[0]], entity[choice_can_route][choice_can_pos[1]]]
        # random.seed(SEED)
        if cal_demand(entity[choice_pos_route]) + DEMAND[edges[0][0]][edges[0][1]] + DEMAND[edges[1][0]][edges[1][1]] <= \
                CAPACITY:
            inverse_rate = random.random()
            if inverse_rate <= ALPAH:
                entity[choice_pos_route].insert(choice_pos_pos, edges[0])
                entity[choice_pos_route].insert(choice_pos_pos + 1, edges[1])
            else:
                entity[choice_pos_route].insert(choice_pos_pos, inverseArc(edges[1]))
                entity[choice_pos_route].insert(choice_pos_pos + 1, inverseArc(edges[0]))
            entity[choice_can_route].remove(edges[0])
            entity[choice_can_route].remove(edges[1])
            break
        # count += 1
        # if count >= 100000:
        #     break
    return entity


# -------------------------------------------------------
# Move operator: Swap(SW)
# -------------------------------------------------------
def SW(entity):
    """
    Swap arbitrary edges
    :param entity:
    :return:
    """
    while True:
        positions = {}
        candidates = {}
        for i in entity.keys():
            positions[i] = []
            candidates[i] = []
            if entity[i][0][0] != DEPOT:
                positions[i] += [0]
            if entity[i][-1][1] != DEPOT:
                positions[i] += [len(entity[i]) - 1]
            if entity[i][0][0] != DEPOT and entity[i][0][1] != entity[i][1][0]:
                candidates[i] += [0]
            for j in range(1, len(entity[i])):
                if entity[i][j - 1][1] != entity[i][j][0]:
                    positions[i] += [j]
                if j < len(entity[i]) - 1 and entity[i][j][0] != entity[i][j - 1][1] and entity[i][j][1] != \
                        entity[i][j + 1][0]:
                    candidates[i] += [j]
                elif j == len(entity[i]) - 1 and entity[i][j - 1][1] != entity[i][j][0] and entity[i][j][1] != DEPOT:
                    candidates[i] += [j]
        while True:
            # random.seed(SEED)
            choice_can_route_1, choice_can_route_2 = [random.randint(1, len(candidates)-1) for _ in range(2)]
            if choice_can_route_1 == choice_can_route_2:
                continue
            elif candidates[choice_can_route_1] and candidates[choice_can_route_2]:
                break
        # random.seed(SEED)
        choice_can_pos_1 = random.randint(0, len(candidates[choice_can_route_1]) - 1)
        choice_can_pos_2 = random.randint(0, len(candidates[choice_can_route_2]) - 1)
        swap_1 = entity[choice_can_route_1][choice_can_pos_1]
        swap_2 = entity[choice_can_route_2][choice_can_pos_2]
        route_copy_1 = copy.deepcopy(entity[choice_can_route_1])
        route_copy_1.remove(swap_1)
        route_copy_2 = copy.deepcopy(entity[choice_can_route_2])
        route_copy_2.remove(swap_2)
        if cal_demand(route_copy_1) + DEMAND[swap_2[0]][swap_2[1]] <= CAPACITY and cal_demand(route_copy_2) + \
                DEMAND[swap_1[0]][swap_1[1]] <= CAPACITY:
            # random.seed(SEED)
            inverse_rate = random.random()
            if 0 <= inverse_rate < ALPAH / 4.:
                route_copy_1.insert(choice_can_pos_1, swap_2)
                route_copy_2.insert(choice_can_pos_2, swap_1)
            elif ALPAH / 4. <= inverse_rate < ALPAH / 2.:
                route_copy_1.insert(choice_can_pos_1, inverseArc(swap_2))
                route_copy_2.insert(choice_can_pos_2, swap_1)
            elif ALPAH / 2. <= inverse_rate < 3 * ALPAH / 2.:
                route_copy_1.insert(choice_can_pos_1, swap_2)
                route_copy_2.insert(choice_can_pos_2, inverseArc(swap_1))
            else:
                route_copy_1.insert(choice_can_pos_1, inverseArc(swap_2))
                route_copy_2.insert(choice_can_pos_2, inverseArc(swap_1))
            entity[choice_can_pos_1 + 1] = copy.deepcopy(route_copy_1)
            entity[choice_can_pos_2 + 1] = copy.deepcopy(route_copy_2)
            break
    return entity


# def fitness(entity):
#     """
#     Calculate the fitness of entities
#     :param entity:
#     :return:
#     """
#     W = {}
#     C = {}
#     P = {}
#     for i in entity.keys():
#         W[i] = 0
#         C[i] = 0
#         W[i] += CAPACITY - cal_demand(entity[i])
#         C[i] += cal_cost(entity[i])
#         W[i] = max(W[i], 0)
#         P[i] = W[i] * C[i]
#     Pw = sum(P.values())
#     return Pw


# -------------------------------------------------------
# Command line control
# -------------------------------------------------------
def command_line(argv):
    try:
        if len(argv) != 5:
            raise getopt.GetoptError("The command line should take 5 arguments(none is given), but only %s input."
                                     % len(argv))
        file_name = argv[0]
        if not os.path.isfile(file_name):
            raise IOError("The carp instance file does not exist.")
        termination = ''
        seed = ''
        opts, args = getopt.getopt(argv[1:], "t:s:")
        for opt, arg in opts:
            if opt == '-t':
                termination = int(arg)
            elif opt == '-s':
                pattern = '^[0-9]+$'
                match = re.findall(pattern, arg)
                if len(match) != 0:
                    seed = long(arg)
                else:
                    seed = arg
        return file_name, termination, seed
    except getopt.GetoptError as err:
        print str(err)
        print 'The argument should be <carp instance file> -t <termination> -s <random seed>'
        sys.exit(2)
    except ValueError as err:
        print(err)
        print 'The argument should be <carp instance file> -t <termination> -s <random seed>'
        sys.exit(2)
    except IOError as err:
        print str(err)
        print 'The argument should be <carp instance file> -t <termination> -s <random seed>'
        sys.exit(2)


if __name__ == "__main__":
    file_name, termination, seed = command_line(sys.argv[1:])
    SEED = seed
    # file_name = 'CARP_samples/egl-s1-A.dat'
    start = time.time()
    G = read_file(file_name)
    pop = population_init()
    sorted_pop_key = sorted(pop[2], key=pop[2].get)
    best_index = sorted_pop_key[0]
    result = ""
    routes = []
    for i in pop[1][best_index].keys():
        routes += [0]+pop[1][best_index][i]+[0]
    for item in routes:
        result += str(item) + ", "
    while (time.time() - start) > termination-1:
        print "s", result[:-2]
        print "q " + str(pop[2][best_index])
        print (time.time() - start)
        quit()
    print "s", result[:-2]
    print "q " + str(pop[2][best_index])

