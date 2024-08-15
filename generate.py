import random
import os
import math

# seeds: int = 1
testset_type: str = "test"

if not os.path.exists(os.path.join("testset", testset_type)):
    os.makedirs(os.path.join("testset", testset_type))


def gen_undirected_graph(n):
    nodes = list(range(n))

    in_graph = [random.choice(nodes)]
    nodes.remove(in_graph[0])

    edges = []
    # ensure the graph is connected
    while nodes:
        rnd, to = random.choice(nodes), random.choice(in_graph)
        nodes.remove(rnd)
        in_graph.append(rnd)

        edges.append((rnd, to) if rnd < to else (to, rnd))

    # add extra edges
    for _ in range(random.randint(n, n)): # FIXME: collision!
        u = random.randint(0, n - 2)
        v = random.randint(u + 1, n - 1)
        if (u, v) not in edges:
            edges.append((u, v))

    # randomly swap u, v
    edges = [(u, v) if random.randint(0, 1) else (v, u) for u, v in edges]
    random.shuffle(edges)
    return edges


for seeds in range(1, 2):
    # random.seed(42)

    # tiny
    # n = random.randint(8, 15)
    # packet_size = random.randint(25, 35)
    # mean = packet_size / random.randint(4, 13)
    # std_dev = mean / 2


    # xsmall
    # n = random.randint(100, 150)
    # packet_size = random.randint(15, 25)
    # mean = packet_size / 2
    # std_dev = mean / 2

    # small
    nodes = random.randint(20, 50)
    packet_num = 1000

    # samples = [max(round(random.gauss(mean, std_dev)), 0) + 1 for _ in range(n)]
    graph = gen_undirected_graph(nodes)
    edges = len(graph)

    src = random.randint(0, nodes)
    dst = random.randint(0, nodes)

    with open(f"testset/{testset_type}/{str(seeds).zfill(3)}.txt", 'w', encoding="utf-8") as f:
        f.write(f"{nodes} {edges} {packet_num}\n")
        for u, v in graph:
            f.write(f"{u} {v} {random.random()}\n")
        for _ in range(packet_num):
            f.write(f"{src} {dst} {100}\n")

# n = 10
# ans = gen_undirected_graph(n)
# print(n, len(ans))
# print("300 0 100")  # TODO: WTF does the second line mean???
# for i, (u, v) in enumerate(ans):
#     # print(f"{u} {v}")  # for https://csacademy.com/app/graph_editor/
#     print(f"{i} {u} {v}")