from abc import ABC, abstractmethod
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import List, Dict
import networkx as nx

def kl_divergence(p, q):
    def new_log(x):
        if x == 0:
            return x
        return math.log(x)
    
    def new_div(x, y):
        if y == 0:
            return 0 if x == 0 else float('inf')
        return x / y

    return p * new_log(new_div(p, q)) + (1 - p) * new_log(new_div(1 - p, 1 - q))

class Packet:
    def __init__(self, src, dst, time):
        self.src: int = src
        self.dst: int = dst
        self.time = time

class Policy(ABC):
    def __init__(self, type: str, graph: nx.Graph) -> None:
        self.type = type
        self.graph = graph
    
    def __repr__(self):
        return f"policy: {self.type}"
    
    @abstractmethod
    def weight_func(self, success: int, attempt: int, tol: float=1e-9):
        pass

    @abstractmethod
    def Jt(self, src: int, dst: int):
        pass

    @abstractmethod
    def choose_best(self, src: int, dst: int):
        pass




class Totoro(Policy):
    def __init__(self, graph: nx.Graph) -> None:
        super().__init__("Totoro", graph)
        self.t = 0
        self.C = 1

    def constraint(self, u, mean_success_rate, attempt):
        return (attempt * kl_divergence(mean_success_rate, u)
                 - self.C * math.log(self.t)) <= 0


    def weight_func(self, success: int, attempt: int, tol: float=1e-9):
        if attempt == 0:
            # if not yet attempted, try it first
            return 0

        mean_success_rate = success / attempt
        
        low, high = mean_success_rate, 1
        
        while high - low > tol:
            mid = (low + high) / 2
            if self.constraint(mid, mean_success_rate, attempt):
                low = mid
            else:
                high = mid

        return 1 / low
    
    def Jt(self, src, dst):
        try:
            # Find the shortest path using Dijkstra's algorithm
            weight_func = lambda u, v, d: d['ETC']
            path = nx.dijkstra_path(self.graph, src, dst, weight=weight_func)
            # Calculate the total cost of the path
            cost = sum(self.graph.edges[(u, v)]['ETC'] for u, v in zip(path, path[1:]))

            return path, cost
        except nx.NetworkXNoPath:
            return None, 0
        
    def choose_best(self, src, dst, t):
        self.t = t
        best_neighbor = []
        best_cost = float('inf')

        for neighbor in self.graph.neighbors(src):
            link = self.graph.edges[(src, neighbor)]
            link['ETC'] = self.weight_func(link['success'], link['attempt'])

        for neighbor in self.graph.neighbors(src):
            path, cost = self.Jt(neighbor, dst)
            total_cost = self.graph.edges[(src, neighbor)]['ETC'] + cost

            if path is None:
                print(f"No path found from {src} to {dst}")
                exit(1)

            if best_cost > total_cost:
                best_cost = total_cost
                best_neighbor = [neighbor]
            elif (best_cost - total_cost) <= 1e-10:
                best_neighbor.append(neighbor)

        return best_neighbor


class Simulator:
    def __init__(self, name: str) -> None:
        self.Policy = name
        self.policy: Policy = None
        self.graph = nx.Graph()
        self.packets: Dict[Packet] = {}
        self.t = 2
    
    def load_sim(self, path: str):
        # load graph and packet sequence from .txt file
        # doesn't check the format
        fp = open(path, 'r')
        itertor = iter(fp)
        
        V, E, packet_num = map(int, next(itertor).split())
        
        # load link and it's prob
        for _ in range(E):
            src, dst, prob = list(map(float, next(itertor).split()))
            src, dst = int(src), int(dst)
            self.graph.add_edge(src, dst, hidden_success_rate=prob, ETC=0, attempt=0, success=0)

        # load packets
        for idx in range(packet_num):
            src, dst, time = list(map(float, next(itertor).split()))
            self.packets[idx] = Packet(src, dst, time)
        
        fp.close()

        # init policy
        if self.Policy == "Totoro":
            self.policy = Totoro(self.graph)
        else:
            print(f"no such policy: {self.Policy}")
            exit(1)

        print(f"load graph with {V} nodes and {E} edges with {packet_num} packets")    

    def simulate(self):

        for idx, packet in self.packets.items():
            src, dst = int(packet.src), int(packet.dst)
            print(f"sending packet {idx} from {src} to {dst}")
            
            while src != dst:
                # print(f"packet pos: {src}")
                best_neighbor = self.policy.choose_best(src, dst, self.t)

                if len(best_neighbor) > 1:
                    best_neighbor = random.choice(best_neighbor)
                else:
                    best_neighbor = best_neighbor[0]
                
                print(f"choose link {src}->{best_neighbor}, ", end="")

                chosen_link = self.graph.edges[(src, best_neighbor)]
                if np.random.random() <= chosen_link['hidden_success_rate']:
                    print("transmission success")
                    chosen_link['success'] += 1
                    src = best_neighbor
                else:
                    print("transmission failed")
                chosen_link['attempt'] += 1

                self.t += 1

            for link, attri in self.graph.edges.items():
                print(f"link {link} {attri}")
            # break

sim = Simulator("Totoro")
sim.load_sim("test.txt")
sim.simulate()
