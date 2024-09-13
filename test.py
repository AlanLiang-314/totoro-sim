from totoro import Simulator
import multiprocessing as mp

def simulate_and_collect(c):
    sim = Simulator("Totoro")
    sim.load_sim("testset/test/003.txt", packet_num=10000)
    sim.reset()
    his = sim.simulate(c=c)
    return his

path_history = [[] for _ in range(4)]  # Adjust size as needed


if __name__ == '__main__':
    for C in [0.1, 0.5, 1, 1.414, 3, 20]:
        with mp.Pool(processes=8) as pool:
            results = pool.map(simulate_and_collect, [C for _ in range(100)])

        regrets = []
        for hist in results:
            # regret_log = []
            regret = 0
            for a, b, c in zip(hist[0], hist[1], hist[2]):
                if a == 1:
                    regret += 0.158-0.119
                    # regret_log.append(regret)
                    continue
                if b == 1:
                    regret += 0.158-0.096
                    # regret_log.append(regret)
                    continue
                if c == 1:
                    regret += 0.158-0.158
                    # regret_log.append(regret)
                    continue
            regrets.append(regret)

        avg_regret = sum(regrets) / len(regrets)
        std_regret = (sum([(r - avg_regret) ** 2 for r in regrets]) / len(regrets)) ** 0.5

        print(f"C={C}, avg_regret={avg_regret}, std_regret={std_regret}")