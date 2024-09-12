from totoro import Simulator
import multiprocessing as mp

def simulate_and_collect(c):
    sim = Simulator("End2End")
    sim.load_sim("testset/test/003.txt", packet_num=1000)
    sim.reset()
    his = sim.simulate(c=c)
    return his

path_history = [[] for _ in range(4)]  # Adjust size as needed


if __name__ == '__main__':

    with mp.Pool(processes=8) as pool:
        results = pool.map(simulate_and_collect, [1.414 for _ in range(50)])

    for his in results:
        for i, h in enumerate(his):
            path_history[i].append(sum(h))

    print(path_history)