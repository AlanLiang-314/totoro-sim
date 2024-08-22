from abc import ABC, abstractmethod
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import List, Dict
import networkx as nx

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
    
    @abstractmethod
    def weight_func(self, success: int, attempt: int, tol: float=1e-9):
        pass

    @abstractmethod
    def Jt(self, src: int, dst: int):
        pass

    @abstractmethod
    def choose_best(self, src: int, dst: int, t: int):
        pass


class Greedy(Policy):
    def __init__(self, graph: nx.Graph) -> None:
        super().__init__("Greedy", graph)
        self.t = 0
        self.C = 1.414
        
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
        return 0
    
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
            best_neighbor = random.choice(best_neighbor)
        else:
            best_neighbor = best_neighbor[0]

        return best_neighbor


class End2End(Policy):
    def __init__(self, graph: nx.Graph) -> None:
        super().__init__("End2End", graph)
        self.t = 0
        self.C = 1.414
        self.first_attempt = True
        self.avaliable_path = [] # hack
        
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
                
        if low == 0:
            return float('inf')

        return 1 / low

    def Jt(self, src, dst):
        return 0
    
    def update_path_status(self, path_id: int, success: bool):
        if self.avaliable_path:
            if success:
                self.avaliable_path[path_id]['success'] += 1
            self.avaliable_path[path_id]['attempt'] += 1
    
    def choose_best(self, src, dst, t, printing=False):
        if self.first_attempt:
            for path in nx.all_simple_paths(self.graph, source=src, target=dst):
                print(f"path: {path}")
                self.avaliable_path.append({'path': path, 'success': 0, 'attempt': 0})
            self.first_attempt = False
            print(f"there are {len(self.avaliable_path)} paths from {src} to {dst}")
        
        self.t = t
        best_path = []
        best_cost = float('inf')
                
        for i, path in enumerate(self.avaliable_path):
            cost = self.weight_func(path['success'], path['attempt'])
            print(f"path {path}, cost {cost}")
            
            if best_cost > cost:
                best_cost = cost
                best_path = [i]
            elif abs(best_cost - cost) <= 1e-10:
                best_path.append(i)

        if len(best_path) > 1:
                best_path = random.choice(best_path)
        else:
            best_path = best_path[0]
            
        return best_path, self.avaliable_path[best_path]['path']


class Totoro(Policy):
    def __init__(self, graph: nx.Graph) -> None:
        super().__init__("Totoro", graph)
        self.t = 0
        self.C = 1.414

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
                best_neighbor = random.choice(best_neighbor)
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
        self.t = 1
    
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

        print(f"load graph with {V} nodes and {E} edges with {packet_num} packets")

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
        path_history = []
        print(f"packet num: {len(self.packets)}")
        
        for idx, packet in self.packets.items():
            src, dst = int(packet.src), int(packet.dst)
            print(f"sending packet {idx} from {src} to {dst}")
            
            best_path_id, best_path = self.policy.choose_best(src, dst, self.t)
            success = True
            print(f"best path: {best_path}")
            
            for u, v in zip(best_path, best_path[1:]):
                chosen_link = self.graph.edges[(u, v)]
                if random.random() > chosen_link['hidden_success_rate']:
                    success = False
                    break
            
            self.t += 1
                
            self.policy.update_path_status(best_path_id, success)
            
            path_string = "->".join(map(str, best_path))
            path_weight = sum(1 / (self.graph.edges[(w, v)]['hidden_success_rate'] + 1e-9) for w, v in zip(best_path, best_path[1:]))
            path_history.append(path_weight)
            print(f"path: {path_string}")
            
        self.plot_hist(path_history)

                    

    def simulate(self):
        # init policy
        if self.Policy == "Totoro":
            self.policy = Totoro(graph=self.graph)
        elif self.Policy == "Greedy":
            self.policy = Greedy(graph=self.graph)
        elif self.Policy == "End2End":
            self.policy = End2End(graph=self.graph)
            self.simulate_end2end()
            return
        else:
            print(f"no such policy: {self.Policy}")
            exit(1)
        
        path_history = []

        for idx, packet in self.packets.items():
            src, dst = int(packet.src), int(packet.dst)
            print(f"sending packet {idx} from {src} to {dst}")

            packet_path = [src]
            
            while src != dst:
                # print(f"packet pos: {src}")
                best_neighbor = self.policy.choose_best(src, dst, self.t)
                
                # print(f"choose link {src}->{best_neighbor}, ", end="")

                chosen_link = self.graph.edges[(src, best_neighbor)]
                if random.random() <= chosen_link['hidden_success_rate']:
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
            path_weight = sum(1 / (self.graph.edges[(w, v)]['hidden_success_rate'] + 1e-9) for w, v in zip(packet_path, packet_path[1:]))
            path_history.append(path_weight)
            
            print(f"path: {path_string}")
            
        self.plot_hist(path_history)
            
    def plot_hist(self, path_history: List[float]):

        src, dst = self.packets[0].src, self.packets[0].dst
        best_path = self.shortest_path(src, dst)
        path_string = "->".join(map(str, best_path))
        print(f"best path: {path_string}")

        window_size = 20
        moving_avg = np.convolve(path_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(path_history, label='Path History', marker='o')

        plt.xlabel('packet num')
        plt.ylabel('Path weight')
        plt.title('Path History and Moving Average')
        plt.legend()

        # Show the plot
        plt.show()

