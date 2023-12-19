import random
import csv
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt

# 遺伝的アルゴリズムの評価関数
def evaluate(individual):
    return sum(individual),

# DEAPの設定
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)  # バイナリ遺伝子の生成
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)  # 遺伝子の長さは100と仮定
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # ビット反転突然変異
toolbox.register("select", tools.selTournament, tournsize=3)


# メイン関数
def main():
    random.seed(42)  # 乱数シードを設定
    generations = 10000  # 世代数
    csv_filename = "ga_results_onemax_populationsize_cxpb_mutpb.csv"
    data = [] # データを格納するリスト

    for _ in range(1000):  # 1000回ループ
        theta = random.uniform(0, 1)  # パラメータθをランダムに生成 # populationsize
        theta2 = random.uniform(0, 1) # cxpb
        theta3 = random.uniform(0, 1) # mutpb
        population_size = (int)(theta * 998) + 2  # 各ループごとにランダムに生成された値
        cxpb = round(theta2, 2) # 交叉率を2つ目のパラメータに設定
        mutpb = round(theta3, 2)
        evaluations_per_theta = []

        for _ in range(10):
            # 集団の初期化
            population = toolbox.population(n=population_size)

            # 遺伝的アルゴリズムの実行
            optimum_found_at_generation = None  # 最適解が見つかった世代を追跡するための変数
            # print(f"theta: {theta}")
            # print(f"population: {population_size}")
            # print(f"theta2: {theta2}")
            # print(f"cxpb: {cxpb}")

            if cxpb + mutpb >1.0:
                mutpb = 1.00 - cxpb

            for gen in range(generations):
                algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, cxpb=cxpb, mutpb=mutpb, ngen=1, stats=None, halloffame=None, verbose=False)
                
                # 最適解の取得
                best_ind = tools.selBest(population, 1)[0]
                # c(θ)の評価回数?を取得
                evaluations = best_ind.fitness.values

                # 最適解が見つかった場合、評価回数を記録してループを終了
                if any(evaluations[0] == 100.0 for ind in population):
                    optimum_found_at_generation = gen + 1
                    evaluations_until_optimum = population_size * (optimum_found_at_generation + 1)
                    break

            if optimum_found_at_generation:
                print(f"最適解が見つかった世代: {optimum_found_at_generation}")
                evaluations_per_theta.append(evaluations_until_optimum)
                # csv_filename = "ga_results_onemax.csv"
                # with open(csv_filename, 'a', newline='') as csv_file:
                #     csv_writer = csv.writer(csv_file)
                #     csv_writer.writerow([population_size, cxpb, evaluations_until_optimum])

            else:
                print("最適解は見つかりませんでした")
                evaluations_per_theta.append(population_size * generations)
                # csv_filename = "ga_results_onemax.csv"
                # with open(csv_filename, 'a', newline='') as csv_file:
                #     csv_writer = csv.writer(csv_file)
                #     csv_writer.writerow([population_size, cxpb, population_size * generations])    

        # 各パラメータにおける評価回数の平均を計算し、データに追加
        average_evaluations = sum(evaluations_per_theta) / len(evaluations_per_theta)
        data.append([int(population_size), float(cxpb), float(mutpb), int(average_evaluations)])

     # print(data)
    np.set_printoptions(suppress=True, precision=2)  # データを表示する前に設定
    data_np = np.array(data)
    # print(data_np)
    data_sorted = data_np[np.argsort(data_np[:,3])]
    print(data_sorted)

    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)

    # 最適解
    data_best_θ1 = np.array(data_sorted[0][0])
    data_best_θ2 = np.array(data_sorted[0][1])
    data_best_θ3 = np.array(data_sorted[0][2])
    # print(data_best)

    # # # FDCの計算
    # dist = [abs((data_sorted[i][j][0] - data_best_θ1)*(data_sorted[i][j][1] - data_best_θ2)) for i,j in range(10)]
    # dist = [np.linalg.norm(abs((data_sorted[i][0] - data_best_θ1)-50)/150, abs(data_sorted[i][1] - data_best_θ2), ord=2) for i in range(10)]
    dist = [np.linalg.norm([(data_sorted[i][0] - data_best_θ1 - 2) / 998, data_sorted[i][1] - data_best_θ2, data_sorted[i][2] - data_best_θ3], ord=2) for i in range(1000)]
    # dist = [np.sqrt((((data_sorted[i][0] - data_best_θ1)-50)/150))*(((data_sorted[i][0] - data_best_θ1)-50)/150)+((data_sorted[i][1] - data_best_θ2)*(data_sorted[i][1] - data_best_θ2)) for i in range(10)]
    print(dist)
    # # sita = [d[0] for d in data]
    # # total_evaluations = [d[1] for d in data]
    # # fdc = sum(a * b for a, b in zip(sita, total_evaluations))

    # # print(f"FDC: {fdc}")
    evaluations = [data_sorted[i][3] for i in range(1000)]

    # VCの計算
    mean_evaluations = np.mean(evaluations)
    std_evaluations = np.std(evaluations)
    vc = std_evaluations / mean_evaluations    

    # 相関係数の計算と表示
    correlation = np.corrcoef(dist, evaluations)[0, 1]
    print(f"相関係数： {correlation}")
    print(f"VC: {vc}")

    # 散布図の描画
    plt.figure(figsize = (8, 6))
    plt.scatter(dist, evaluations)
    plt.xlabel('distance')
    plt.ylabel('evaluations')
    plt.title('fdc_population size')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
