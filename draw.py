import csv
from collections import Counter
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
import ast

# 读取对应算法数据 
def read_algorithm_data(file_path, alg):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['algorithm'] == alg:
                return row
    return None

# 读取所有算法数据
def read_data(read_algorithm_data, file_path, fieldnames, algs, data_set):
    for alg in algs:
        data = read_algorithm_data(file_path, alg)
        if data is not None:
            for fieldname in fieldnames:
                if fieldname == 'algorithm':
                    data_set[alg][fieldname] = alg
                else:
                    data_str = data.get(fieldname, "")
                    try:
                        data_list = ast.literal_eval(data_str) if data_str else []
                    except (ValueError, SyntaxError):
                        data_list = []
                    data_set[alg][fieldname] = data_list
        else:
            print(f"No data of algorithm {alg}!!!")
            continue

#  mode depth combination制作饼图
def polt_pie(algs, data_set, output_dir):
    for alg in algs:
        if alg in ['Basic','Residual','Dense']:
            continue   
        keys = list(data_set[alg].keys())[1:]  # 从第二个键开始
        titles = [alg+'_mode',alg+'_depth']  # 假设从第二个键开始的标题
        for key, title in zip(keys, titles):
            element_counts = Counter(data_set[alg][key])
            plt.figure()
            plt.pie(element_counts.values(), labels=element_counts.keys(), autopct='%1.1f%%')
        # plt.axis('equal')  # 确保饼图是圆形的
            plt.title(title)
            # plt.legend(title=alg)  # 添加标签
        # plt.tight_layout()
            plt.savefig(os.path.join(output_dir, alg +'_'+ title + '.png'))
            plt.close()

# 制作对比的折线图
def plot_multiple_algorithms(fieldnames, algs, data_set, output_dir):
    for y_label in fieldnames[4:]:
        plt.figure()
        for alg in algs:
            plt.plot(data_set[alg][y_label], label=alg)
        plt.xlabel('Time slot')
        plt.ylabel(y_label)
        plt.legend()
        plt.title(y_label)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        formatter.set_useLocale(True)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'All_' + y_label + '.png'))
        plt.close()

# 对同一个算法的不同指标进行对比
def self_contrast(data_set, output_dir, choose_fieldnames, choose_algs_fieldnames):
    for alg in choose_algs_fieldnames:
        for couple in choose_fieldnames:
            plt.figure()
            for metrics in couple: 
                plt.plot(data_set[alg][metrics], label=metrics)
            plt.xlabel('Time slot')
            plt.ylabel('Value')
            plt.legend()
            plt.title(couple[0])
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
            formatter.set_useLocale(True)
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, alg + '_'+ couple[0] + '.png'))
            plt.close()

# 图片输出路径
output_dir = os.path.join('.', 'data', 'picture') 
# 指标
fieldnames = [
    'algorithm', 'mode','depth','combination','h0_reward', 'h1_reward', 'reward', 'bpp', 'mse',
    'cover_scores', 'message_density', 'uiqi', 'rs', 'psnr', 'ssim', 'consumption','generated_scores',
    ]
# 算法们
algorithms = ['DQN','HRL','Basic',
            'Residual','Dense',]
# 文件路径
file_path = os.path.join('data', 'data.csv')  
if __name__ == '__main__':
    # 选择对比的算法
    choose_algs = [0,1,2,3,4,]
    algs = [algorithms[alg] for alg in choose_algs]
    # 数据
    data_set = {alg: {fieldname: [] for fieldname in fieldnames} for alg in algs}
    read_data(read_algorithm_data, file_path, fieldnames, algs, data_set)
    # 画图
    # mode depth combination饼图
    polt_pie(algs, data_set, output_dir)
    # 对比折线图
    plot_multiple_algorithms(fieldnames, algs, data_set, output_dir)
    # 对比指标
    choose_fieldnames = [['uiqi','uiqi_v'],['rs','rs_v']]
    # 对比算法
    choose_algs_fieldnames = ['HRL_e1','HRL_e1']
    # 自身指标对比
    # self_contrast(data_set, output_dir, choose_fieldnames, choose_algs_fieldnames)