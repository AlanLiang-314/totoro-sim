import random
import os
import math

# seeds: int = 1
testset_type: str = "dense_connected"

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
    for _ in range(random.randint(4 * n, 8 * n)): # FIXME: collision!
        u = random.randint(0, n - 2)
        v = random.randint(u + 1, n - 1)
        if (u, v) not in edges:
            edges.append((u, v))

    # randomly swap u, v
    edges = [(u, v) if random.randint(0, 1) else (v, u) for u, v in edges]
    random.shuffle(edges)
    return edges


for seeds in range(1, 101):
    random.seed(seeds)

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
    n = random.randint(250, 650)
    packet_size = random.randint(25, 65)
    mean = packet_size / random.randint(3, 5) - 1
    std_dev = (random.random() * 4)

    samples = [max(round(random.gauss(mean, std_dev)), 0) + 1 for _ in range(n)]
    samples[0] = 0
    ans = gen_undirected_graph(n)
    agg_start_time = round(math.log2(n) * 30)
    dis_start_time = agg_start_time * 2
    data_trans_time = agg_start_time * 3
    sim_time = data_trans_time * 2

    with open(f"testset/{testset_type}/{str(seeds).zfill(3)}.txt", 'w', encoding="utf-8") as f:
        f.write(f"{n} {len(ans)} {packet_size}\n{sim_time} {0} {agg_start_time}\n{dis_start_time} {data_trans_time}\n")
        for i, weight in enumerate(samples):
            f.write(f"{i} {weight}\n")
        for i, (u, v) in enumerate(ans):
            f.write(f"{i} {u} {v}\n")

# n = 10
# ans = gen_undirected_graph(n)
# print(n, len(ans))
# print("300 0 100")  # TODO: WTF does the second line mean???
# for i, (u, v) in enumerate(ans):
#     # print(f"{u} {v}")  # for https://csacademy.com/app/graph_editor/
#     print(f"{i} {u} {v}")