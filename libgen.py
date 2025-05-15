import re
import pprint
import random
import numpy as np
import csv
import argparse
import os
import sys
from numpy.linalg import matrix_rank, det, eig, svd, norm
from scipy.sparse import lil_matrix
import time
from scipy.sparse import save_npz, csr_matrix
from scipy.sparse import find
from scipy.sparse import triu
import itertools

def parse_gate_info(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    if verbose_lib == "yes":
        # ファイル内容を確認
        print("File content:\n", data)
        
    # Regular expression patterns to match the gate information
    #gate_pattern = re.compile(r'([A-Z_]+)\nlabel:\n\s*\[(.*?)\]\nbias:\n\s*\[(.*?)\]\nweight:\n((?:\s*\[.*?\]\n?)+)\nenergy:\n\s*\[(.*?)\]\n')
    gate_pattern = re.compile(r'([A-Z_]+)\nlabel:\n\s*\[(.*?)\]\nbias:\n\s*\[(.*?)\]\nweight:\n((?:\s*\[.*?\]\n?)+)\nenergy:\n\s*\[(.*?)\]\noutput:\n\s*\[(.*?)\]\n')
    weight_pattern = re.compile(r'\[\s*(.*?)\s*\]')

    gates = {}
    for match in gate_pattern.finditer(data):
        gate_name = match.group(1)
        labels = match.group(2).split(',')
        bias = list(map(int, match.group(3).split(',')))
        
        weight_str = match.group(4)
        weights = [list(map(int, weight.split(','))) for weight in weight_pattern.findall(weight_str)]
        
        energy = int(match.group(5))

        output = int(match.group(6))
    
        gates[gate_name] = {
            'label': [label.strip() for label in labels],
            'bias': bias,
            'weight': weights,
            'energy': energy,
            'output': output
        }

    if verbose_lib == "yes":
        # デバッグ情報を表示
        print("Parsed gates:\n", gates)
        
    return gates

# ゲート情報をランダムに読み出し、行列を更新する関数
def update_matrix_with_gate(matrix, gate, scale, gate_index):
    #indices = random.sample(range(added_gates), len(gate['bias']）-1)
    indices = random.sample(range(gate_index), len(gate['bias'][:-1]))
    indices.append(gate_index)
    indices.sort()
    print(indices)
    for i, idx in enumerate(indices):
        matrix[idx, idx] += scale * gate['bias'][i]
    for i in range(len(gate['weight'])):
        for j in range(len(gate['weight'][i])):
            matrix[indices[i], indices[j]] += scale * gate['weight'][i][j]
    
    return matrix

##############################
def run_generation(graph_type, size, scale, seed, force, feature, verbose_lib, scale_mode="fixed"):
    
    # ユーザーが選択したゲート群
    if graph_type == 'HZA':
        user_selected_gates = ['FA', 'NOT', 'BUF']  #h=0, J=+-
        dir_path = 'HZA'
    elif graph_type == 'HZN':
        user_selected_gates = ['FAXX', 'NOT']  #h=0, J=-
        dir_path = 'HZN'
    elif graph_type == 'HNN':
        user_selected_gates = ['NOR','XOR'] #h=-, J=-
        dir_path = 'HNN'
    elif graph_type == 'HPN': 
        user_selected_gates = ['NAND','XNOR'] #h=+, J=-
        dir_path = 'HPN'
    elif graph_type == 'HAA':
        user_selected_gates = ['BUF', 'NOT', 'OR', 'NOR', 'AND', 'NAND', 'XOR', 'XNOR', 'FA'] # all gates
        dir_path = 'HAA'
    elif graph_type == 'nor_xor_fa_buf':
        user_selected_gates = ['BUF', 'NOR', 'XOR','FA'] 
        dir_path = 'nor_xor_fa_buf'
    elif graph_type == 'nand_xnor_fa_buf':
        user_selected_gates = ['BUF', 'NAND', 'XNOR','FA'] 
        dir_path = 'nand_xnor_fa_buf'
    elif graph_type == 'xor_nor':
        user_selected_gates = ['XOR'] 
        dir_path = 'xor_nor'
    elif graph_type == 'xor_or':
        user_selected_gates = ['XOR_OR'] 
        dir_path = 'xor_or'
    elif graph_type == 'xor_nand':
        user_selected_gates = ['XOR_NAND'] 
        dir_path = 'xor_nand'
    elif graph_type == 'xor_and':
        user_selected_gates = ['XOR_AND'] 
        dir_path = 'xor_and'
    elif graph_type == 'xnor_nor':
        user_selected_gates = ['XNOR_NOR'] 
        dir_path = 'xnor_nor'
    elif graph_type == 'xnor_or':
        user_selected_gates = ['XNOR_OR'] 
        dir_path = 'xnor_or'
    elif graph_type == 'xnor_nand':
        user_selected_gates = ['XNOR'] 
        dir_path = 'xnor_nand'
    elif graph_type == 'xnor_and':
        user_selected_gates = ['XNOR_AND'] 
        dir_path = 'xnor_and'
    elif graph_type == 'nand':
        user_selected_gates = ['NAND'] 
        dir_path = 'nand'
    elif graph_type == 'and':
        user_selected_gates = ['AND'] 
        dir_path = 'and'
    elif graph_type == 'nor':
        user_selected_gates = ['NOR'] 
        dir_path = 'nor'
    elif graph_type == 'or':
        user_selected_gates = ['OR'] 
        dir_path = 'or'
    
    print("graph_type", graph_type)
    # 2次元行列のサイズ（例：4x4行列）
    matrix_size = size
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(f'./matrix/{dir_path}', exist_ok=True)
    
    filename = f'./matrix/{dir_path}/{dir_path}_N{matrix_size}_S{scale}_matrix.npz'
    
    # Check if the file exists
    if os.path.exists(filename) and force == 'no':
        print(f"File {filename} already exists. Exiting the program.")
        sys.exit()
    
    file_path = './HamiltonianList_2-body.txt'
    gates = parse_gate_info(file_path)
    #pprint.pprint(gates)
    
    # 乱数シードを固定
    random_seed = seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    start_time = time.time()
    
    # 最終行列の初期化
    #final_matrix = np.zeros((matrix_size, matrix_size))
    # 初期行列を疎行列で生成
    final_matrix = lil_matrix((matrix_size, matrix_size))
    total_energy = 0
    added_gates = 0
    gate_index = 2
    nonzero_ratio = 0

    while True:
        available_gates = [gate for gate in user_selected_gates if gate in gates]
        if not available_gates:
            print("No matching gates found in the provided list.")
            break

        random_gate_name = random.choice(available_gates)
        gate_info = gates[random_gate_name]

        gate_index += gate_info['output']
        if gate_index >= matrix_size:
            print(f"Breaking the loop as gate_index ({gate_index}) >= matrix_size ({matrix_size})")
            break

        # --- スケールの決定 ---
        if scale_mode == "fixed":
            current_scale = scale
        elif scale_mode == "uniform":
            current_scale = random.uniform(0.1, scale)  # 例：0.1～scale の範囲でランダム
        elif scale_mode == "normal":
            gauss_rnd = random.gauss(scale, 0.2*scale)
            current_scale = max(min(gauss_rnd, scale*2), 0.0)
        
        total_energy += current_scale * gate_info['energy']
        final_matrix = update_matrix_with_gate(final_matrix, gate_info, current_scale, gate_index)
        added_gates += 1
    
    nonzero_ratio = final_matrix.count_nonzero() / (matrix_size * matrix_size)
    print(f"Size: {size}, graph_Type: {graph_type}, Non-zero ratio: {nonzero_ratio:.2%}")
    print("Min energy :", total_energy)
          
    # LIL形式のfinal_matrixをCSR形式に変換
    final_matrix_sparse = final_matrix.tocsr()
    
    # 結果を表示
    print("Final Sparse Matrix:\n", final_matrix_sparse)
    print(f"追加したゲートの数: {added_gates}")
    print(f"Non-zero ratio: {nonzero_ratio:.2%}")
    
    # 終了時間を記録
    end_time = time.time()
    
    # 計算時間を表示
    elapsed_time = end_time - start_time
    print(f"計算時間: {elapsed_time:.4f} 秒")
    
    #######################################
    # 対角成分の抽出
    diagonal_elements = final_matrix_sparse.diagonal()
    print('h:', diagonal_elements)
    
    # 非対角成分の抽出（上三角部分のみ）
    non_diagonal_elements = triu(final_matrix_sparse, k=1).data
    print('J:', non_diagonal_elements)
    
    # 対角成分の統計量
    diag_mean = np.mean(diagonal_elements)
    diag_std = np.std(diagonal_elements)
    diag_min = np.min(diagonal_elements)
    diag_max = np.max(diagonal_elements)
    diag_median = np.median(diagonal_elements)
    diag_sum = np.sum(diagonal_elements)
    diag_amean = np.mean(abs(diagonal_elements))
    diag_amin = np.min(abs(diagonal_elements))
    diag_amax = np.max(abs(diagonal_elements))
    
    # 非対角成分の統計量
    non_diag_mean = np.mean(non_diagonal_elements)
    non_diag_std = np.std(non_diagonal_elements)
    non_diag_min = np.min(non_diagonal_elements)
    non_diag_max = np.max(non_diagonal_elements)
    non_diag_median = np.median(non_diagonal_elements)
    non_diag_sum = np.sum(non_diagonal_elements)
    non_diag_amean = np.mean(abs(non_diagonal_elements))
    non_diag_amin = np.min(abs(non_diagonal_elements))
    non_diag_amax = np.max(abs(non_diagonal_elements))
    
    ##############################
    # 各ノードの接続数（非対角成分のみ）を計算
    J_sparse = final_matrix_sparse.copy()
    J_sparse.setdiag(0)  # 対角成分を0に設定
    nonzero_counts = J_sparse.getnnz(axis=1)  # 各行のノンゼロ要素数を計算
    
    # Step 3: 最小値、平均値、最大値、標準偏差を計算
    min_nonzero = nonzero_counts.min()
    mean_nonzero = nonzero_counts.mean()
    max_nonzero = nonzero_counts.max()
    std_nonzero = nonzero_counts.std()
    
    # 結果を出力
    print(f"各ノードの接続数の最小値: {min_nonzero}")
    print(f"各ノードの接続数の平均値: {mean_nonzero:.2f}")
    print(f"各ノードの接続数の最大値: {max_nonzero}")
    print(f"各ノードの接続数の標準偏差: {std_nonzero:.2f}")
    
    # ノンゼロ要素の割合を計算
    total_elements = final_matrix_sparse.shape[0] * final_matrix_sparse.shape[1]
    non_diag_matrix = triu(final_matrix_sparse, k=1)
    nonzero_elements_non_diag = non_diag_matrix.nnz * 2  # 非対角成分は上三角のみ計算し倍にする
    nonzero_ratio_non_diag = nonzero_elements_non_diag / (total_elements - final_matrix_sparse.shape[0])
    print('total_non_diagonal_elements:', total_elements - final_matrix_sparse.shape[0])
    print('total_non_diagonal_nonzero_elements', nonzero_elements_non_diag)
    print('nonzero_ratio (non diag):', nonzero_ratio_non_diag)
    
    nonzero_elements_diag = np.count_nonzero(final_matrix_sparse.diagonal())
    nonzero_ratio_diag = nonzero_elements_diag / final_matrix_sparse.shape[0]
    print('total_diagonal_elements:', final_matrix_sparse.shape[0])
    print('total_diagonal_nonzero_elements', nonzero_elements_diag)
    print('nonzero_ratio (diag):', nonzero_ratio_diag)
    
    # 対角成分と非対角成分の統計量
    diagonal_elements = final_matrix_sparse.diagonal()
    non_diagonal_elements = non_diag_matrix.data
    
    # 統計量計算
    diag_mean = np.mean(diagonal_elements)
    diag_std = np.std(diagonal_elements)
    diag_min = np.min(diagonal_elements)
    diag_max = np.max(diagonal_elements)
    diag_median = np.median(diagonal_elements)
    diag_sum = np.sum(diagonal_elements)
    
    non_diag_mean = np.mean(non_diagonal_elements)
    non_diag_std = np.std(non_diagonal_elements)
    non_diag_min = np.min(non_diagonal_elements)
    non_diag_max = np.max(non_diagonal_elements)
    non_diag_median = np.median(non_diagonal_elements)
    non_diag_sum = np.sum(non_diagonal_elements)
    
    print("\nDiagonal Elements Statistics:")
    print("Mean Value:", diag_mean)
    print("Standard Deviation:", diag_std)
    print("Minimum Value:", diag_min)
    print("Maximum Value:", diag_max)
    print("Median Value:", diag_median)
    print("Sum Value:", diag_sum)
    
    print("\nNon-Diagonal Elements Statistics:")
    print("Mean Value:", non_diag_mean)
    print("Standard Deviation:", non_diag_std)
    print("Minimum Value:", non_diag_min)
    print("Maximum Value:", non_diag_max)
    print("Median Value:", non_diag_median)
    print("Sum Value:", non_diag_sum)
    
    #####################絶対値最大値による正規化########################
    max_aH = np.max(np.abs(final_matrix_sparse.data))
    anH_sparse = final_matrix_sparse / max_aH
    
    diagonal_elements_anH = anH_sparse.diagonal()
    non_diagonal_elements_anH = triu(anH_sparse, k=1).data
    
    diag_mean_anH = np.mean(diagonal_elements_anH)
    diag_std_anH = np.std(diagonal_elements_anH)
    
    non_diag_mean_anH = np.mean(non_diagonal_elements_anH)
    non_diag_std_anH = np.std(non_diagonal_elements_anH)
    
    # 非ゼロ要素のみを使って統計量を計算
    mean_anH = np.mean(anH_sparse.data)  # 非ゼロ要素の平均
    std_anH = np.std(anH_sparse.data)    # 非ゼロ要素の標準偏差
    
    print('anH:',anH_sparse)
    print("Mean Value (diag) anH:", diag_mean_anH)
    print("Standard Deviation (diag) anH:", diag_std_anH)
    print("Mean Value (non diag) anH:", non_diag_mean_anH)
    print("Standard Deviation (non diag) anH:", non_diag_std_anH)
    print('Mean Value anH"', mean_anH)
    print('Standard Deviation anH:',std_anH)
    
    ##################### Zスコア正規化 ########################
    # 非ゼロ要素に基づいて疎行列をZスコア正規化
    mean_H = np.mean(final_matrix_sparse.data)  # 非ゼロ要素の平均
    std_H = np.std(final_matrix_sparse.data)    # 非ゼロ要素の標準偏差
    
    znH_sparse = final_matrix_sparse.copy()    # 元の行列をコピー
    znH_sparse.data = (znH_sparse.data - mean_H) / std_H  # Zスコア正規化を適用
    
    # 正規化後の対角成分と非対角成分を抽出
    diagonal_elements_znH = znH_sparse.diagonal()
    non_diagonal_elements_znH = triu(znH_sparse, k=1).data
    
    # 統計量の計算
    diag_mean_znH = np.mean(diagonal_elements_znH)
    diag_std_znH = np.std(diagonal_elements_znH)
    
    non_diag_mean_znH = np.mean(non_diagonal_elements_znH)
    non_diag_std_znH = np.std(non_diagonal_elements_znH)
    
    mean_znH = np.mean(znH_sparse.data)
    std_znH = np.std(znH_sparse.data)
    
    # 結果を出力
    print('anH:',znH_sparse)
    print("Mean Value (diag) znH:", diag_mean_znH)
    print("Standard Deviation (diag) znH:", diag_std_znH)
    print("Mean Value (non diag) znH:", non_diag_mean_znH)
    print("Standard Deviation (non diag) znH:", non_diag_std_znH)
    print('Mean Value znH"', mean_znH)
    print('Standard Deviation znH:',std_znH)
    
    ########################################
    # .npz形式で疎行列を保存
    sparse_filename = f'./matrix/{dir_path}/{dir_path}_N{matrix_size}_S{scale}_{scale_mode}_matrix.npz'
    
    # .npz形式で保存
    save_npz(sparse_filename, final_matrix_sparse)
    print(f"Sparse matrix saved to {sparse_filename}")
    
    # 非ゼロ要素のインデックスと値を取得
    rows, cols, values = find(final_matrix_sparse)
    
    # CSVに出力
    matrix_filename = f'./matrix/{dir_path}/{dir_path}_N{matrix_size}_S{scale}_{scale_mode}_matrix.csv'
    with open(matrix_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row', 'col', 'value'])  # ヘッダー行（オプション）
        for r, c, v in zip(rows, cols, values):
            writer.writerow([r, c, v])
    
    # Save the statistics to a separate CSV file
    stats_filename = f'./matrix/{dir_path}/{dir_path}_N{matrix_size}_S{scale}_{scale_mode}_feature.csv'
    with open(stats_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["graph_Type", dir_path])
        writer.writerow(["Matrix size", matrix_size])
        writer.writerow(["Number of adding", added_gates])
        #writer.writerow(["Target nonzero ratio", target_ratio])
        writer.writerow(["Nonzero ratio", nonzero_ratio])
        writer.writerow(["Min energy", total_energy])
        writer.writerow(["Min connection", min_nonzero])
        writer.writerow(["Avg connection", mean_nonzero])
        writer.writerow(["Max connection", max_nonzero])
        writer.writerow(["Std connection", std_nonzero])
        writer.writerow(["Mean (diag) H", diag_mean])
        writer.writerow(["Standard Deviation (diag) H", diag_std])
        writer.writerow(["aMean (diag) H", diag_amean])
        writer.writerow(["Min (diag) H", diag_min])
        writer.writerow(["Max (diag) H", diag_max])
        writer.writerow(["Mean (non diag) H", non_diag_mean])
        writer.writerow(["Standard Deviation (non diag) H", non_diag_std])
        writer.writerow(["aMean (non diag) H", non_diag_amean])
        writer.writerow(["Min (non diag) H", non_diag_min])
        writer.writerow(["Max (non diag) H", non_diag_max])
        writer.writerow(["Mean H", mean_H])
        writer.writerow(["Standard Deviation H", std_H])
        writer.writerow(["Mean (diag) anH", diag_mean_anH])
        writer.writerow(["Standard Deviation (diag) anH", diag_std_anH])
        writer.writerow(["Mean (non diag) anH", non_diag_mean_anH])
        writer.writerow(["Standard Deviation (non diag) anH", non_diag_std_anH])
        writer.writerow(["Mean anH", mean_anH])
        writer.writerow(["Standard Deviation anH", std_anH])
        writer.writerow(["Mean (diag) znH", diag_mean_znH])
        writer.writerow(["Standard Deviation (diag) znH", diag_std_znH])
        writer.writerow(["Mean (non diag) znH", non_diag_mean_znH])
        writer.writerow(["Standard Deviation (non diag) znH", non_diag_std_znH])
        writer.writerow(["Mean znH", mean_znH])
        writer.writerow(["Standard Deviation znH", std_znH])
    print(f"Statistics saved to {stats_filename}")

def main():
    # Parameters for library gen
    size_list = [100, 250, 500, 1000, 2500]
    type_list = ["HAA", "HZA", "HZN", "HNN", "HPN"]
    scale_list = [0.5, 1.0, 2.0]
    scale_mode_list = ["normal", "uniform", "fixed"]
    seed = [42]
    force = ["no"]
    feature = ["no"]
    verbose_lib = ["no"]

    # 全組み合わせを生成
    param_combinations = itertools.product(type_list, size_list, scale_list, scale_mode_list, seed, force, feature, verbose_lib)

    # 各パラメータセットでシミュレーションを実行
    for graph_type, size, scale, scale_mode, seed, force, feature, verbose_lib in param_combinations:
        print(f"Running simulation with graph_type={graph_type}, size={size}, scale={scale}, seed={seed}, force={force}")
        run_generation(graph_type, size, scale, seed, force, feature, verbose_lib, scale_mode)

if __name__ == "__main__":
    main()