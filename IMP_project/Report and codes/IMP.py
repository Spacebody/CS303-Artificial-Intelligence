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
import numpy as np

__author__ = "Yilin Zheng"
__version__ = "6.5.8"


# parameter
p = 0.001
R = 1000
INFTY = sys.maxsize


# functions
def read_file(network_file):
    """
    read network data and selected seed from files
    """
    first_line = network_file.readline()
    data = str.split(first_line)
    node_num = int(data[0])
    nodes = range(1, node_num + 1)
    edge_num = int(data[1])
    network = defaultdict(dict)
    Dv = defaultdict(int)
    data_lines = network_file.readlines()
    for line in data_lines:
        data = str.split(line)
        head = int(data[0])
        tail = int(data[1])
        Dv[head] += 1
        prob = float(data[2])
        network[head][tail] = prob
    return nodes, edge_num, network, Dv


def IC(network, seeds):
    """
    independent cascade model
    """
    if not seeds:
        return 0
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
    if not seeds:
        return 0
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
    parser.add_argument("-k", "--seed_size", metavar="<predefined size of the seed set>", required=True,
                        type=int, help="a positive integer which is the predefined size of the seeds")
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
    seed_size = args.seed_size
    seeds = []
    nodes, edge_num, network, Dv = read_file(args.network)
    random.seed(args.random_seed)
    if args.termination:
        count = 0.
        iterations = 0
        if args.model == "IC":
            start = time.clock()
            while time.clock() - start < args.time_budget - 1.5:
                seeds = CELF2(network, nodes, seed_size, Dv, args.model)
                count += IC(network, seeds)
                iterations += 1
        elif args.model == "LT":
            start = time.clock()
            while time.clock() - start < args.time_budget - 1.5:
                seeds = CELF2(network, nodes, seed_size, Dv, args.model)
                count += LT(network, seeds)
                iterations += 1
        print("Algorithm applied: CEFL")
        print("model: {}".format(args.model))
        print("seeds: {}".format(seeds))
        print("result: {}".format(count / iterations))
    else:
        seeds = CELF2(network, nodes, seed_size, Dv, args.model)
        result = default_evaluation(network, seeds, args.model)
        print("Algorithm applied: CEFL")
        print("model: {}".format(args.model))
        print("seeds: {}".format(seeds))
        print("result: {}".format(result))

    # Other algorithms, uncommented the following codes for trial
    # print("MaxDegree:")
    # seeds = MaxDegree(network, seed_size)
    # print(seeds)
    # print(default_evaluation(network, seeds, "IC"))
    # print(default_evaluation(network, seeds, "LT"))
    
    # print("DegreeDiscount:")
    # seeds = DegreeDiscount(network, nodes, seed_size, Dv)
    # print(default_evaluation(network, seeds, "IC"))
    # print(default_evaluation(network, seeds, "LT"))
    
    # print("CELF:")
    # seeds = CELF(network, nodes, seed_size, Dv, args.model)
    # print(seeds)
    # print(default_evaluation(network, seeds, args.model))
    
    # print("GeneralGreedy:")
    # seeds = GeneralGreedy(network, nodes, seed_size, Dv)
    # print(seeds)
    # print(default_evaluation(network, seeds, args.model))


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


# MaxDegree
def MaxDegree(network, seed_size):
    """
    By select the first k nodes with largest degree
    """
    degree_record = defaultdict(int)
    for head in network:
        degree_record[head] = len(network[head].keys())
    seeds = sorted(degree_record, key=degree_record.get, reverse=True)[:seed_size]
    return seeds


# DegreeDiscount
def DegreeDiscount(network, nodes, seed_size, Dv):
    global p
    seeds = []
    dead_node = []
    Tv = defaultdict(int)
    DDv = defaultdict(float)
    for node in nodes:
        if node not in Dv.keys():
            dead_node.append(node)
        DDv[node] = Dv[node]
    for i in range(seed_size):
        u = max(Dv, key=Dv.get)
        seeds.append(u)
        Dv.pop(u)
        for v in network[u]:
            if v not in Dv.keys():
                for x in seeds:
                    if x in network[v]:
                        Tv[v] += 1
                DDv[v] = Dv[v] - 2 * Tv[v] - \
                    (Dv[v] - Tv[v]) * Tv[v] * p
    print("dead node:", dead_node)
    print(seeds)
    return seeds


# CELF
def CELF(network, nodes, seed_size, Dv, model):
    seeds_UC = LazyForward(network, nodes, seed_size, "UC", Dv)
    seeds_CB = LazyForward(network, nodes, seed_size, "CB", Dv)
    if default_evaluation(network, seeds_UC, model) >= default_evaluation(network, seeds_CB, model):
        return seeds_UC
    else:
        return seeds_CB


def LazyForward(network, nodes, seed_size, compute_type, Dv):
    seeds = []
    V = nodes[:]
    delta = defaultdict(float)
    cur = defaultdict(bool)
    for node in nodes:
        if node in Dv.keys():
            delta[node] = INFTY
            cur[node] = False
        else:
            V.remove(node)
    while True:
        for node in V:
            if compute_type == "UC":
                node_star = max(delta, key=delta.get)
            elif compute_type == "CB":
                node_star = max(delta, key=lambda k: delta[k] / (len(seeds) + 1))
            if cur[node]:
                seeds.append(node_star)
                if len(seeds) == seed_size:
                    return seeds
                delta.pop(node_star)
            else:
                delta[node] = delta_R(network, seeds, node)
                cur[node] = True
    return seeds


def delta_R(network, seeds, node):
    seeds_be = seeds[:]
    seeds_af = seeds[:]
    seeds_af.append(node)
    if not seeds_be:
        return default_evaluation(network, seeds_af, "IC") 
    delta = max(0, default_evaluation(network, seeds_af, "IC") - default_evaluation(network, seeds_be, "IC"))
    return delta


# GeneralGreedy
def GeneralGreedy(network, nodes, seed_size, Dv):
    seeds = []
    global R
    s_v = defaultdict(float)
    for i in range(seed_size):
        for node in nodes:
            s_v[node] = 0
            if node not in seeds and node in Dv.keys():
                for i in range(R):
                    s_v[node] += IC(network, seeds + [node]) - IC(network, seeds)
                s_v[node] /= R
        seeds.append(max(s_v, key=s_v.get))
    return seeds


def CELF2(network, nodes, seed_size, Dv, model):
    seeds = []
    R = 1000
    s_v = defaultdict(float)
    if model == "IC":
        while len(seeds) < seed_size:
            if seeds:
                reevaluate_node = max(s_v, key=s_v.get)
                s_v[reevaluate_node] = 0
                for i in range(R):
                    s_v[reevaluate_node] += IC(network, seeds + [reevaluate_node]) - IC(network, seeds)
                s_v[reevaluate_node] /= R
                new_node = max(s_v, key=s_v.get)
                if new_node == reevaluate_node:
                    seeds.append(new_node)
                    s_v.pop(new_node)
                else:
                    continue
            else:
                for node in nodes:
                    s_v[node] = 0
                    if node in Dv.keys():
                        for i in range(R):
                            s_v[node] += IC(network, seeds + [node])
                        s_v[node] /= R
                first_seed = max(s_v, key=s_v.get)
                seeds.append(first_seed)
                s_v.pop(first_seed)
                # s_v = sorted(s_v, key=s_v.get, reverse=True)
    elif model == "LT":
        while len(seeds) < seed_size:
            if seeds:
                reevaluate_node = max(s_v, key=s_v.get)
                s_v[reevaluate_node] = 0
                for i in range(R):
                    s_v[reevaluate_node] += LT(network, seeds +
                                               [reevaluate_node]) - LT(network, seeds)
                s_v[reevaluate_node] /= R
                new_node = max(s_v, key=s_v.get)
                if new_node == reevaluate_node:
                    seeds.append(new_node)
                    s_v.pop(new_node)
                else:
                    continue
            else:
                for node in nodes:
                    s_v[node] = 0
                    if node in Dv.keys():
                        for i in range(R):
                            s_v[node] += LT(network, seeds + [node])
                        s_v[node] /= R
                first_seed = max(s_v, key=s_v.get)
                seeds.append(first_seed)
                s_v.pop(first_seed)
    return seeds


if __name__ == "__main__":
    main()
 
