from abc import ABC, abstractmethod
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import List, Dict
import networkx as nx
import scipy.stats as stats


def kl_divergence(p, q):
    # kl divergence, but can deal with [0, 1]
    
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
    
    def constraint(self, u, mean_success_rate, attempt):
        return (attempt * kl_divergence(mean_success_rate, u)
                 - self.C * math.log(self.t)) <= 0
    
    # @abstractmethod
    def weight_func(self, success: int, attempt: int, tol: float=1e-9):
        # print(f"self.t {self.t}, self.C: {self.C}")
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
        if low == 0:
            return 0
        else:
            return 1 / low

    @abstractmethod
    def choose_best(self, src: int, dst: int, t: int):
        pass


class Greedy(Policy):
    def __init__(self, graph: nx.Graph, c=1.414) -> None:
        super().__init__("Greedy", graph)
        self.t = 0
        self.C = c
    
    def choose_best(self, src, dst, t, printing=False):
        self.t = t
        best_neighbor = []
        best_cost = float('inf')

        for neighbor in self.graph.neighbors(src):
            link = self.graph.edges[(src, neighbor)]
            link['ETC'] = self.weight_func(link['success'], link['attempt'])
            
            if best_cost > link['ETC']:
                best_cost = link['ETC']
                best_neighbor = [neighbor]
            elif abs(best_cost - link['ETC']) <= 1e-10:
                best_neighbor.append(neighbor)
                
            if printing:
                print(f"best neighbor {best_neighbor}, best cost {best_cost}")
                
        if len(best_neighbor) > 1:
            best_neighbor = np.random.choice(best_neighbor)
        else:
            best_neighbor = best_neighbor[0]

        return best_neighbor


class End2End(Policy):
    def __init__(self, graph: nx.Graph, c = 1.414) -> None:
        super().__init__("End2End", graph)
        self.t = 0
        self.C = c
        self.first_attempt = True
        self.avaliable_path = [] # hack
        
    def update_path_status(self, path_id: int, success: bool):
        if self.avaliable_path:
            if success:
                self.avaliable_path[path_id]['success'] += 1
            self.avaliable_path[path_id]['attempt'] += 1
    
    def choose_best(self, src, dst, t, printing=False):
        if self.first_attempt:
            for path in nx.all_simple_paths(self.graph, source=src, target=dst):
                # print(f"path: {path}")
                self.avaliable_path.append({'path': path, 'success': 0, 'attempt': 0})
            self.first_attempt = False
            # print(f"there are {len(self.avaliable_path)} paths from {src} to {dst}")
        
        self.t = t
        best_path = []
        best_cost = float('inf')
                
        for i, path in enumerate(self.avaliable_path):
            cost = self.weight_func(path['success'], path['attempt'])
            # print(f"path {path}, cost {cost}")
            
            if best_cost > cost:
                best_cost = cost
                best_path = [i]
            elif abs(best_cost - cost) <= 1e-10:
                best_path.append(i)

        if len(best_path) > 1:
                best_path = np.random.choice(best_path)
        else:
            best_path = best_path[0]
            
        return best_path, self.avaliable_path[best_path]['path']
    
    def choose_best_2(self, src, dst, t, printing=False):
        if self.first_attempt:
            for path in nx.all_simple_paths(self.graph, source=src, target=dst):
                print(f"path: {path}")
                self.avaliable_path.append({'path': path, 'success': 0, 'attempt': 0})
            self.first_attempt = False
            print(f"there are {len(self.avaliable_path)} paths from {src} to {dst}")
        
        self.t = t
        costs = []
                        
        for i, path in enumerate(self.avaliable_path):
            cost = self.weight_func(path['success'], path['attempt'])
            print(f"path {path}, cost {cost}")
            costs.append(cost)
            
        costs = np.array(costs)
        softmax_probs = np.exp(-costs) / np.sum(np.exp(-costs))
        best_path = np.random.choice(range(len(costs)), p=softmax_probs)
            
        return best_path, self.avaliable_path[best_path]['path']



class Totoro(Policy):
    def __init__(self, graph: nx.Graph, c=1.414) -> None:
        super().__init__("Totoro", graph)
        self.t = 0
        self.C = c

    def Jt(self, src, dst):
        try:
            # Find the shortest path using Dijkstra's algorithm
            weight_func = lambda u, v, d: d['ETC']
            path = nx.dijkstra_path(self.graph, src, dst, weight=weight_func)
            # Calculate the total cost of the path
            cost = sum(self.graph.edges[(u, v)]['ETC'] for u, v in zip(path, path[1:]))

            return path, cost
        except nx.NetworkXNoPath:
            return None, float('inf')
        
    def choose_best(self, src, dst, t, printing=False):
        self.t = t
        best_neighbor = []
        best_cost = float('inf')

        for neighbor in self.graph.neighbors(src):
            link = self.graph.edges[(src, neighbor)]
            link['ETC'] = self.weight_func(link['success'], link['attempt'])

        for neighbor in self.graph.neighbors(src):
            path, cost = self.Jt(neighbor, dst)
            total_cost = self.graph.edges[(src, neighbor)]['ETC'] + cost
            if printing:
                print(f"path {path}, cost {total_cost}")

            if path is None:
                print(f"No path found from {src} to {dst}")
                exit(1)

            
            if best_cost > total_cost:
                best_cost = total_cost
                best_neighbor = [neighbor]
            elif abs(best_cost - total_cost) <= 1e-10:
                best_neighbor.append(neighbor)
                
            if printing:
                print(f"best neighbor {best_neighbor}, best cost {best_cost}")
        
        if len(best_neighbor) > 1:
                best_neighbor = np.random.choice(best_neighbor)
        else:
            best_neighbor = best_neighbor[0]

        return best_neighbor


class Simulator:
    def __init__(self, name: str) -> None:
        self.Policy = name
        self.policy: Policy = None
        # self.graph = nx.Graph()
        self.graph = nx.DiGraph()
        self.packets: Dict[int, Packet] = {}
        self.V: int = 0
        self.E: int = 0
        self.start: int = None
        self.end: int = None
        self.packet_num: int = 0
        self.t = 1
        # np.random.seed(42)
    
    def load_sim(self, path: str, packet_num: int = 100):
        # load graph and packet sequence from .txt file
        # doesn't check the format
        fp = open(path, 'r')
        itertor = iter(fp)
        
        self.V, self.E, self.packet_num, self.start, self.end = map(int, next(itertor).split())
        self.packet_num = packet_num
        
        # load link and it's prob
        for _ in range(self.E):
            src, dst, prob = list(map(float, next(itertor).split()))
            src, dst = int(src), int(dst)
            self.graph.add_edge(src, dst, hidden_success_rate=prob, ETC=0, attempt=0, success=0)

        # load packets
        # for idx in range(packet_num):
        #     src, dst, time = list(map(float, next(itertor).split()))
        #     self.packets[idx] = Packet(src, dst, time)
        
        fp.close()

        # print(f"load graph with {self.V} nodes and {self.E} edges with {self.packet_num} packets")

    def shortest_path(self, src, dst):
            weight_func = lambda u, v, d: 1 / (d['hidden_success_rate'] + 1e-9)
            path = nx.dijkstra_path(self.graph, src, dst, weight=weight_func)
            return path
        
    def reset(self):
        for edge in self.graph.edges.values():
            edge['success'] = 0
            edge['attempt'] = 0
            edge['ETC'] = 0
            self.t = 1
            
    def simulate_end2end(self):
    
        
        paths = list(nx.all_simple_paths(self.graph, source=self.start, target=self.end))

        path_history = [[] for i in range(len(paths))]
        
        # print(f"packet num: {len(self.packets)}")
        
        for idx in range(self.packet_num):
            src, dst = self.start, self.end
            # src, dst = int(packet.src), int(packet.dst)
            # print(f"sending packet {idx} from {src} to {dst}")
            
            best_path_id, best_path = self.policy.choose_best(src, dst, self.t)
            success = True
            # print(f"best path: {best_path}")
            
            for u, v in zip(best_path, best_path[1:]):
                chosen_link = self.graph.edges[(u, v)]
                if np.random.random() > chosen_link['hidden_success_rate']:
                    success = False
                    break
            
            self.t += 1
                
            self.policy.update_path_status(best_path_id, success)
            
            # path_string = "->".join(map(str, best_path))
            # print(f"path: {path_string}")
            
            for i, path in enumerate(paths):
                path_history[i].append(1 if best_path == path else 0)
            
        # print("iter done")
        self.plot_attempt(paths, path_history)
        # start = 500
        # end = 1000
        # for path, ceil, floor in zip(paths, path_ceil, path_floor):
        #     # color = next(plt.gca()._get_lines.prop_cycler)['color']  # 獲取下一個顏色
        #     fill_between_obj = plt.fill_between(range(len(ceil[start:end])), ceil[start:end], floor[start:end], alpha=0.3, label=f"path {'->'.join(map(str, path))}")  # 填充 ceil 和 floor 之間的區域
        #     fill_color = fill_between_obj.get_facecolor()[0]
        #     fill_color = (fill_color[0], fill_color[1], fill_color[2], 1)
        #     plt.plot(ceil[start:end], color= fill_color)
        #     plt.plot(floor[start:end], color = fill_color)
        # plt.legend()
        # plt.show()

        return path_history


                    

    def simulate(self, c):
        # init policy
        if self.Policy == "Totoro":
            self.policy = Totoro(graph=self.graph, c=c)
        elif self.Policy == "Greedy":
            self.policy = Greedy(graph=self.graph, c=c)
        elif self.Policy == "End2End":
            self.policy = End2End(graph=self.graph, c=c)
            path_history = self.simulate_end2end()
            return path_history
        else:
            print(f"no such policy: {self.Policy}")
            exit(1)
        
        paths = list(nx.all_simple_paths(self.graph, source=self.start, target=self.end))

        path_history = [[] for i in range(len(paths))]



        for idx in range(self.packet_num):
            src, dst = self.start, self.end
            # src, dst = int(packet.src), int(packet.dst)
            # print(f"sending packet {idx} from {src} to {dst}")

            packet_path = [src]
            
            while src != dst:
                # print(f"packet pos: {src}")
                best_neighbor = self.policy.choose_best(src, dst, self.t)
                
                # print(f"choose link {src}->{best_neighbor}, ", end="")

                chosen_link = self.graph.edges[(src, best_neighbor)]
                if np.random.random() <= chosen_link['hidden_success_rate']:
                    # print("transmission success")
                    chosen_link['success'] += 1
                    packet_path.append(best_neighbor)
                    src = best_neighbor
                else:
                    pass
                    # print("transmission failed")
                chosen_link['attempt'] += 1

                self.t += 1
            
            path_string = "->".join(map(str, packet_path))            
            # print(f"path: {path_string}")
            
            for i, path in enumerate(paths):
                path_history[i].append(1 if packet_path == path else 0)

            
        # shortest_path = self.shortest_path(src, dst)
        # print(shortest_path)
        # self.plot_attempt(paths, path_history)
        # print("iter done")
        return path_history
        
        
        
    def plot_graph(self):
        for edge in self.graph.edges:
            attri = self.graph.edges[edge]
            if attri['success'] > 0:
                diff = abs(attri['attempt'] / attri['success'] - attri['hidden_success_rate'])
            else:
                diff = 1
            
                
            
            
    def plot_attempt(self, paths, path_history):
        for path, history in zip(paths, path_history):
            path_history_cumsum = np.cumsum(history)
            prob = 1
            for v1, v2 in zip(path, path[1:]):
                prob *= self.graph.edges[(v1, v2)]['hidden_success_rate']
                
            plt.plot(path_history_cumsum, label=f"path {'->'.join(map(str, path))}, prob {prob:.3f}")
            
        plt.xlabel('packet num')
        plt.ylabel('attempt')
        plt.title('Path History')
        
        plt.legend()
        plt.show()
