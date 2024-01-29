import random
import csv
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools, algorithms
# from vcopt import vcopt # pip install vcopt が必要


# NKランドスケープの要素数とKを設定
N = 15  # 要素数
K = 2   # 各要素の相互作用数

random.seed(25) # NK参照表のスコアのための乱数

window_size = K + 1
bit_list = ['{:b}'.format(i).zfill(window_size) for i in range(2**window_size)]
score_list = np.random.rand(2**window_size)
table = dict(zip(bit_list, score_list)) # NK参照表

gene_list = ['{:b}'.format(i).zfill(N) for i in range(2**N)]
# best_score,best_geneを全探索で求める？
best_gene = ''
best_score = 0
gene_list_double = np.core.defchararray.add(gene_list, gene_list)
for gene in gene_list_double:
    score = 0
    for i in range(0, N):
      score += table[gene[i:i+window_size]]
    score /= N
    if score > best_score:
        best_gene = gene # 最適解(*2?)
        best_score = score # 最適適応度

best_gene = best_gene[:N]
print(f"最適解：{best_gene}")
print(f"最適適応度：{best_score}")

def gene_score(para): # para：遺伝子配列. 個体を受け取ってそのスコアを返す関数
    gene = ''.join(map(str, para))

    gene_double = gene + gene
    score = 0
    for i in range(0, N):
        score += table[gene_double[i:i+window_size]]
    score /= N

    return score,

# DEAPの設定
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)  # バイナリ遺伝子の生成
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, N)  # 遺伝子の長さは100と仮定
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", gene_score)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # ビット反転突然変異
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)
    generations = 1000 # 世代数
    csv_filename = "ga_results_NK_cxpb.csv"
    data = []
    n = 1

    for _ in range(1000):
        theta = random.uniform(0, 1)
        population_size = 180
        cxpb = theta * 6 /10 + 0.3
        cxpb = round(cxpb, 3)        
        evaluations_per_theta = []

        for _ in range(10):

            # 集団の初期化
            population = toolbox.population(n=population_size)

            # 遺伝的アルゴリズムの実行
            optimum_found_at_generation = None # 最適解が見つかった世代を追跡するための変数
            if cxpb > 0.9:
                mutpb = 1.00 - cxpb
            else:
                mutpb = 0.01   
            for gen in range(generations):
                algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, cxpb=cxpb, mutpb=mutpb, ngen=1, stats=None, halloffame=None, verbose=False)

                # 最適解の取得
                best_ind = tools.selBest(population, 1)[0]
                #最適解の評価回数を取得
                evaluations = best_ind.fitness.values
                # print(f"evaluations{evaluations}")
            
                # 最適解が見つかった場合、評価回数を記録してループを終了
                if any(evaluations[0] == best_score for ind in population):
                    optimum_found_at_generation = gen + 1
                    evaluations_until_optimum = population_size * (optimum_found_at_generation + 1)
                    break

            if optimum_found_at_generation:
              if(n%100==0):
                print(f"{n}最適解が見つかった世代: {optimum_found_at_generation}")
                # print(f"最適解：{best_ind}")
              evaluations_per_theta.append(evaluations_until_optimum)
              n = n + 1
            else:
                print(f"{n} 最適解は見つかりませんでした")
                evaluations_per_theta.append(population_size * generations)
                n = n + 1

        average_evaluations = sum(evaluations_per_theta) / len(evaluations_per_theta)
        # average_evaluations = np.median(evaluations_per_theta)
        data.append([cxpb, average_evaluations])

    np.set_printoptions(suppress = True, precision = 2)
    data_np = np.array(data)
    data_sorted = data_np[np.argsort(data_np[:,1])]
    print(data_sorted)

    with open(csv_filename, 'w', newline = '') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data_sorted)

    # 最適解
    data_best = np.array(data_sorted[0][0])
    diff = max((data_sorted[0][0]-0.3),(0.9-data_sorted[0][0]))

    # FDCの計算
    dist = [abs(data_sorted[i][0] - data_best) / diff for i in range(1000)]
    print(dist)
    evaluations = [data_sorted[i][1] for i in range(1000)]

    # VCの計算
    mean_evaluations = np.mean(evaluations)
    std_evaluations = np.std(evaluations)
    vc = std_evaluations / mean_evaluations

    # 相関係数の計算と表示
    correlation = np.corrcoef(dist, evaluations)[0, 1]
    print(f"相関係数: {correlation}")
    print(f"VC: {vc}")

    plt.figure(figsize = (8, 6))
    plt.scatter(dist, evaluations)     
    plt.xlabel('distance')
    plt.ylabel('evaluations')
    plt.title('fdc_cxpb')
    plt.grid(True)
    plt.show() 

if __name__ == "__main__":
    main()      
