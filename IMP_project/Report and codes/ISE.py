from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
import time
from threading import Timer
# import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np

__author__ = "Yilin Zheng"
__version__ = "3.3.0"

# G = nx.DiGraph()


# functions
def read_file(network_file, seeds):
    """
    read network data and selected seed from files
    """
    # e = []
    first_line = network_file.readline()
    data = str.split(first_line)
    node_num = int(data[0])
    nodes = range(1, node_num + 1)
    edge_num = int(data[1])
    network = defaultdict(dict)
    data_lines = network_file.readlines()
    for line in data_lines:
        data = str.split(line)
        head = int(data[0])
        tail = int(data[1])
        prob = float(data[2])
        network[head][tail] = prob
        # e += [(data[0], data[1], float(data[2]))]
    # G.add_weighted_edges_from(e)
    # nx.draw(G)
    # plt.show()
    seeds_lines = seeds.readlines()
    seeds = []
    for line in seeds_lines:
        seeds.append(int(str.split(line)[0]))
    return nodes, edge_num, network, seeds


def IC(network, seeds):
    """
    independent cascade model
    """
    active_node = seeds[:]
    node_queue = active_node[:]
    while node_queue:
        head = node_queue.pop(0)
        for tail in network[head]:
            if tail not in active_node:
                prob = random.random()
                if prob <= network[head][tail]:
                    active_node.append(tail)
                    node_queue.append(tail)
    return len(active_node)


def LT(network, seeds):
    """
    linear threshold model
    """
    active_node = seeds[:]
    node_queue = active_node[:]
    prob_record = defaultdict(float)
    node_threshold = defaultdict(float)
    while node_queue:
        head = node_queue.pop(0)
        for tail in network[head]:
            if tail not in active_node:
                if not node_threshold[tail]:
                    node_threshold[tail] = random.random()
                prob_record[tail] += network[head][tail]
                if tail not in active_node and prob_record[tail] >= node_threshold[tail]:
                    active_node.append(tail)
                    node_queue.append(tail)
    return len(active_node)


def command_line():
    parser = argparse.ArgumentParser(description='This program is used to evaluate the \
                                    performance of IMP algorithms by processing either one of the two basic diffusion \
                                    models on the given network with the given seeds.')
    parser.add_argument("-i", "--network", metavar="<social network>", required=True,
                        type=open, help="the absolute path of the social network file")
    parser.add_argument("-s", "--seeds", metavar="<seed set>", required=True,
                        type=open, help="the absolute path of the seed set file")
    parser.add_argument("-m", "--model", metavar="<diffusion model>",
                        required=True, help="only IC or LT")
    parser.add_argument("-b", "--termination", metavar="<termination type>", required=True,
                        type=(lambda x: x == '1'), help="only 0 or 1, 0 to use default conditions while 1 to use time budget")
    parser.add_argument("-t", "--time_budget", metavar="<time budget>", required=True,
                        type=int, help="positive number which indicates the running time in seconds allowed")
    parser.add_argument("-r", "--random_seed", metavar="<random seed>", required=True,
                        type=int, help="seed for random")
    return parser.parse_args()


def main():
    args = command_line()
    nodes, edge_num, network, seeds = read_file(args.network, args.seeds)
    random.seed(args.random_seed)
    if args.termination:
        count = 0.
        iterations = 0   
        if args.model == "IC":
            start = time.clock()
            while time.clock() - start < args.time_budget:
                count += IC(network, seeds)
                iterations += 1
        elif args.model == "LT":
            start = time.clock()
            while time.clock() - start < args.time_budget:
                count += LT(network, seeds)
                iterations += 1
        print("model: {}".format(args.model))
        print("result: {}".format(count / iterations))
    else:
        result = default_evaluation(network, seeds, args.model)
        print("model: {}".format(args.model))
        print("result: {}".format(result))


def default_evaluation(network, seeds, model=None):
    if model == "IC":
        count_IC = 0.
        for i in range(10000):
            count_IC += IC(network, seeds)
        return count_IC / 10000
    elif model == "LT":
        count_LT = 0.
        for i in range(10000):
            count_LT += LT(network, seeds)
        return count_LT / 10000


if __name__ == "__main__":
    main()

