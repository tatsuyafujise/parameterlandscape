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
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 50)  # 遺伝子の長さは100と仮定
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # ビット反転突然変異
toolbox.register("select", tools.selTournament, tournsize=3)


# メイン関数
def main():
    random.seed(60)  # 乱数シードを設定
    generations = 10000  # 世代数
    population_size = 500
    csv_filename = "ga_results_onemax_cxpb*mutpb.csv"
    data = [] # データを格納するリスト
    heatdict = {}
    for key in range(60):
      heatdict[key] = []
    n = 1

    for _ in range(1000):  # 1000回ループ
        theta = random.uniform(0, 1)  # パラメータθをランダムに生成
        theta2 = random.uniform(0, 1)
        cxpb = theta * 6 / 10 + 0.3
        cxpb = round(cxpb, 3)
        mutpb = theta2 * 0.099 +0.001
        mutpb = round(mutpb, 5) # 交叉率を2つ目のパラメータに設定
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

            if mutpb + cxpb > 1.0:
                cxpb = 1.00 - mutpb
            # else:
            #     cxpb = 0.6

            for gen in range(generations):
                algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, cxpb=cxpb, mutpb=mutpb, ngen=1, stats=None, halloffame=None, verbose=False)
                
                # 最適解の取得
                best_ind = tools.selBest(population, 1)[0]
                # c(θ)の評価回数?を取得
                evaluations = best_ind.fitness.values

                # 最適解が見つかった場合、評価回数を記録してループを終了
                if any(evaluations[0] == 50.0 for ind in population):
                    optimum_found_at_generation = gen + 1
                    evaluations_until_optimum = population_size * (optimum_found_at_generation + 1)
                    break

            if optimum_found_at_generation:
                if(n%100==0):
                    print(f"{n}最適解が見つかった世代: {optimum_found_at_generation}")
                evaluations_per_theta.append(evaluations_until_optimum)
                n = n + 1
                # csv_filename = "ga_results_onemax.csv"
                # with open(csv_filename, 'a', newline='') as csv_file:
                #     csv_writer = csv.writer(csv_file)
                #     csv_writer.writerow([population_size, cxpb, evaluations_until_optimum])

            else:
                print(f"{n}最適解は見つかりませんでした")
                evaluations_per_theta.append(population_size * generations)
                n = n + 1
                # csv_filename = "ga_results_onemax.csv"
                # with open(csv_filename, 'a', newline='') as csv_file:
                #     csv_writer = csv.writer(csv_file)
                #     csv_writer.writerow([population_size, cxpb, population_size * generations])    

        # 各パラメータにおける評価回数の平均を計算し、データに追加
        average_evaluations = np.median(evaluations_per_theta)
        heatdict.setdefault((10*(int)(10*(cxpb-0.31))+(int)(100*(mutpb-0.0011))), []).append(average_evaluations)
        data.append([float(cxpb), float(mutpb), int(average_evaluations)])

     # print(data)
    np.set_printoptions(suppress=True, precision=2)  # データを表示する前に設定
    data_np = np.array(data)
    # print(data_np)
    data_sorted = data_np[np.argsort(data_np[:,2])]
    print(data_sorted)
    for i in range(60):
          heatdict[i] = np.mean(heatdict[i])
          # print(f"{i}: {heatdict[i]}")
    zz = np.array(list(heatdict.values()))
    # zz = np.nan_to_num(zz, nan = 300000)
    zz = zz.reshape(6, 10) 
    # zz = np.array2string(zz, separator=', ', formatter={'float_kind': lambda x: '{: .1f}'.format(x)})
    # zz = np.fromstring(zz, sep=', ')
    print(zz)
    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)

    # 最適解
    data_best_θ1 = np.array(data_sorted[0][0])
    data_best_θ2 = np.array(data_sorted[0][1])
    # print(data_best)

    # # # FDCの計算
    # dist = [abs((data_sorted[i][j][0] - data_best_θ1)*(data_sorted[i][j][1] - data_best_θ2)) for i,j in range(10)]
    # dist = [np.linalg.norm(abs((data_sorted[i][0] - data_best_θ1)-50)/150, abs(data_sorted[i][1] - data_best_θ2), ord=2) for i in range(10)]
    dist = [np.linalg.norm([(data_sorted[i][0] - data_best_θ1) * 10 / 6, (data_sorted[i][1] - data_best_θ2) / 0.099], ord=2) for i in range(1000)]
    # dist = [np.sqrt((((data_sorted[i][0] - data_best_θ1)-50)/150))*(((data_sorted[i][0] - data_best_θ1)-50)/150)+((data_sorted[i][1] - data_best_θ2)*(data_sorted[i][1] - data_best_θ2)) for i in range(10)]
    print(dist)
    # # sita = [d[0] for d in data]
    # # total_evaluations = [d[1] for d in data]
    # # fdc = sum(a * b for a, b in zip(sita, total_evaluations))

    # # print(f"FDC: {fdc}")
    evaluations = [data_sorted[i][2] for i in range(1000)]

     # VCの計算
    mean_evaluations = np.mean(evaluations)
    std_evaluations = np.std(evaluations)
    vc = std_evaluations / mean_evaluations    

    # 相関係数の計算と表示
    correlation = np.corrcoef(dist, evaluations)[0, 1]
    print(f"相関係数： {correlation}")
    print(f"VC: {vc}")

    Fig = plt.figure()
    ax1 = Fig.add_subplot(2,1,1)   # 散布図
    ax1.scatter(dist, evaluations)
    ax1.set_xlabel('distance')
    ax1.set_ylabel('evaluations')
    ax1.set_title('fdc_cxpb vs mutpb')
    ax1.grid(True)

    ax2 = Fig.add_subplot(2,1,2)    # ヒートマップ
    X = np.arange(0, 10)
    Y = np.arange(3, 10)
    mappable = ax2.pcolor(zz, edgecolors='k', linewidths=2, cmap='nipy_spectral_r')  # edgecolors, linewidths, cmap を追加
    plt.colorbar(mappable, ax=ax2)    # x = heatdict.keys % 10
    # x = heatdict.keys % 10
    # y = heatdict.keys / 10
    # ax2.pcolor(X,Y,zz)

    # ラベルやタイトルの設定
    ax2.set_xlabel('Mutation Probability')
    ax2.set_ylabel('Crossover Probability')
    ax2.set_title('Heatmap')

    # グリッドの表示
    # plt.grid(True)
    Fig.tight_layout()
    # 表示
    plt.show()

if __name__ == "__main__":
    main()

    
