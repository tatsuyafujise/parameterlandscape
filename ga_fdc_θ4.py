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

# main関数内でトーナメントサイズを変更可能にする
def main():
    random.seed(42)  # 乱数シードを設定
    generations = 10000  # 世代数
    csv_filename = "ga_results_onemax_populationsize_cxpb_mutpb.csv"
    data = [] # データを格納するリスト

    # トーナメントサイズの設定
    tournament_size = 3  # デフォルトのトーナメントサイズ

    for _ in range(1000):  # 1000回ループ
        # ...（以前のコードと同じ）

        for _ in range(10):
            # 集団の初期化
            population = toolbox.population(n=population_size)

            # トーナメントサイズの設定
            toolbox.register("select", tools.selTournament, tournsize=tournament_size)

            # 遺伝的アルゴリズムの実行
            optimum_found_at_generation = None  # 最適解が見つかった世代を追跡するための変数

            # EAの設定時にトーナメントサイズを渡すためのlambda関数を定義
            eaMuPlusLambda = algorithms.eaMuPlusLambda
            eaMuPlusLambda.func_globals["toolbox"] = toolbox
            eaMuPlusLambda.func_globals["mu"] = population_size
            eaMuPlusLambda.func_globals["lambda_"] = population_size
            eaMuPlusLambda.func_globals["cxpb"] = cxpb
            eaMuPlusLambda.func_globals["mutpb"] = mutpb
            eaMuPlusLambda.func_globals["ngen"] = 1
            eaMuPlusLambda.func_globals["stats"] = None
            eaMuPlusLambda.func_globals["halloffame"] = None
            eaMuPlusLambda.func_globals["verbose"] = False
            eaMuPlusLambda.func_globals["toolbox"].register("select", tools.selTournament, tournsize=tournament_size)

            if cxpb + mutpb > 1.0:
                mutpb = 1.00 - cxpb

            for gen in range(generations):
                eaMuPlusLambda(population)

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
               
