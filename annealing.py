#!/usr/bin/env python3
"""
Annealing simulation script refactored with a main() entry‑point and
command‑line presets such as --quick and --normal.

Usage examples
--------------
Quick sanity run:
    python annealing_sim.py --quick

Normal (default) run:
    python annealing_sim.py --normal   # or simply omit, same as default

You can still override individual parameters:
    python annealing_sim.py --quick --trial 5 --cycle 500

"""
import argparse
import itertools
import math
import os
import random
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###############################################################
# ---------------- Core algorithmic functions ----------------#
###############################################################

def annealing(vertex, tau, I0_min, I0_max, beta, nrnd, h_vector, J_matrix,
              spin_vector, Itanh_ini, rand_type, cycle, control, algorithm,
              mean_range, stall_prop):
    """Single annealing run returning the last state and histories."""
    Itanh = Itanh_ini
    spin_history = []
    energy_history = []
    I0_history = []

    I_vector_list = deque()

    # initial I0 (cycle == 1)
    I0 = I0_min + (I0_max - I0_min) * math.log(1 + tau * (1 - 1) / (cycle - 1)) / math.log(1 + tau)
    for t in range(cycle * recool_cycle):
        for _ in range(tau):
            # schedule for I0
            if control == 0:
                I0 = I0_min + (I0_max - I0_min) * math.log(
                    1 + tau * ((t % cycle) - 1) / (cycle - 1)
                ) / math.log(1 + tau)
            if t % cycle == 0 and t > 0:
                I0 = I0_min + (I0_max - I0_min) * math.log(1 + tau * (1 - 1) / (cycle - 1)) / math.log(1 + tau)
                if verbose_ann == "yes":
                    print("recool")
            if verbose_ann == "yes":
                print(f"Cycle {t}: I0 = {I0}")

            # main update per algorithm
            if algorithm == 0:  # pSA
                rnd = generate_random(algorithm, rand_type, len(spin_vector))
                I_vector = h_vector + np.dot(J_matrix, spin_vector)
                Itanh = np.tanh(I0 * I_vector) + nrnd * rnd
                spin_vector = np.where(Itanh >= 0, 1, -1)

            elif algorithm == 1:  # SSA
                rnd = generate_random(algorithm, rand_type, len(spin_vector))
                rnd = np.where(rnd == 0, -1, 1)
                I_vector = h_vector + np.dot(J_matrix, spin_vector) + (nrnd * rnd)
                Itanh = fun_Itanh(Itanh, I_vector, I0)
                spin_vector = np.where(Itanh >= 0, 1, -1)

            elif algorithm in (2, 4):  # TApSA
                rnd = generate_random(algorithm, rand_type, len(spin_vector))
                I_vector = h_vector + np.dot(J_matrix, spin_vector)
                if t >= mean_range:
                    if I_vector_list:
                        I_vector_list.popleft()
                    I_vector_list.append(I_vector)
                else:
                    I_vector_list.append(I_vector)
                I_vector = np.mean(I_vector_list, axis=0)
                Itanh = np.tanh(I0 * I_vector) + nrnd * rnd
                spin_vector = np.where(Itanh >= 0, 1, -1)

            elif algorithm in (3, 5):  # SpSA
                rnd = generate_random(algorithm, rand_type, len(spin_vector))
                I_vector = h_vector + np.dot(J_matrix, spin_vector)
                Itanh = np.tanh(I0 * I_vector) + nrnd * rnd
                random_indices = np.random.choice(vertex, size=np.int32(vertex * (1 - stall_prop)), replace=False)
                spin_vector[random_indices] = np.where(Itanh[random_indices] >= 0, 1, -1)

            I0_history.append(I0)
            spin_history.append(spin_vector.copy())
            energy_history.append(energy_calculate(h_vector, J_matrix, spin_vector, "no"))

    return spin_vector, spin_history, energy_history, I0_history


def fun_Itanh(Itanh, I_vector, I0):
    discriminant = Itanh + I_vector
    Itanh = np.where(discriminant >= I0, I0, discriminant)
    Itanh = np.where(discriminant < -I0, -I0, Itanh)
    return Itanh


def generate_random(algorithm, rand_type, size):
    if algorithm in (0, 2, 3, 4, 5):
        if rand_type == 0:
            return 2.0 * np.random.rand(size, 1) - 1.0
        elif rand_type == 1:
            return 0.1 * np.random.poisson(10, (size, 1)) - 1.0
    elif algorithm == 1:
        return np.random.randint(0, 2, (size, 1))  # 0 or 1


def cut_calculate(G_matrix, spin_vector):
    spin_vector_reshaped = spin_vector.ravel()
    upper_triangle = np.triu_indices(len(spin_vector), k=1)
    cut_val = np.sum(
        G_matrix[upper_triangle] * (1 - np.outer(spin_vector_reshaped, spin_vector_reshaped)[upper_triangle])
    )
    return int(cut_val / 2)


def energy_calculate(h_vector, J_matrix, spin_vector, show):
    h_energy = np.sum(h_vector * spin_vector)
    J_energy = np.sum(spin_vector * np.dot(J_matrix, spin_vector)) / 2
    if show == "yes":
        print("h_energy : ", h_energy)
        print("J_energy : ", J_energy)
    return -(J_energy + h_energy)


def get_graph(vertex, lines):
    G_matrix = np.zeros((vertex, vertex), int)
    for line_text in lines:
        i, j, weight = map(int, line_text.split())
        G_matrix[i - 1, j - 1] = weight
    return G_matrix + G_matrix.T


def read_and_process_file(file_path):
    with open(file_path, 'r') as f:
        vertex = int(f.readline().strip())
        optimal_value = int(f.readline().strip())
        lines = f.readlines()
    G_matrix = get_graph(vertex, lines)
    return vertex, G_matrix, -G_matrix, optimal_value


def save_results(spin_history, energy_history):
    Path("./history").mkdir(exist_ok=True)
    energy_history_df = pd.DataFrame({"Energy": energy_history})
    energy_history_df.to_csv("./history/energy_history.csv", index_label="Step")

    spin_history_matrix = np.hstack(spin_history)
    spin_history_df = pd.DataFrame(spin_history_matrix.T)
    spin_history_df.to_csv(
        "./history/spin_history.csv",
        index_label="Step",
        header=[f"Node_{i}" for i in range(spin_history_matrix.shape[0])],
    )
    print("Histories saved.")


def plot_results(energy_history, spin_history, I0_history, node_index=0):
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history, label="Energy")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy vs Steps During Annealing")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(I0_history, label="I0")
    plt.xlabel("Step")
    plt.ylabel("I0")
    plt.title("I0 vs Steps During Annealing")
    plt.legend()
    plt.grid()
    plt.show()

    node_spin_history = [spin[node_index, 0] for spin in spin_history]
    plt.figure(figsize=(10, 6))
    plt.plot(node_spin_history, label=f"Node {node_index}")
    plt.xlabel("Step")
    plt.ylabel("Spin State (-1 or 1)")
    plt.title("Spin State Over Steps for Node")
    plt.legend()
    plt.grid()
    plt.show()


def save_summary(data, file_path="./result/result.csv"):
    Path("./result").mkdir(exist_ok=True)
    header = [
        'Algorithm', 'Graph', 'Number of nodes', 'Scale', 'Scale mode', 'Number of cycles', 'tau', 'Trials', 'Optimal value',
        'Mean value', 'Min value', 'Max value', 'Normalized mean energy', 
        'Mean flips', 'Min flips', 'Max flips',
        'Gamma', 'Delta', 'rand_type',
        'Mean range', 'Stall probability'
    ]
    write_header = not Path(file_path).is_file()
    with open(file_path, 'a', newline='') as csvfile:
        writer = pd.ExcelWriter
        import csv
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)


def get_graph_library(vertex, file_path):
    J_matrix = np.zeros((vertex, vertex))
    data = pd.read_csv(file_path, header=0)
    for _, row in data.iterrows():
        J_matrix[int(row['row']), int(row['col'])] = row['value']
    h_vector = np.copy(J_matrix.diagonal().reshape(vertex, 1))
    if verbose_ann == "yes":
        print("h=", h_vector)
    print("minh=", np.min(h_vector))
    print("maxh=", np.max(h_vector))

    for i in range(min(J_matrix.shape[0], J_matrix.shape[1])):
        J_matrix[i, i] = 0

    if verbose_ann == "yes":
        print("J=", J_matrix)
    print("minJ=", np.min(J_matrix))
    print("maxJ=", np.max(J_matrix))

    return h_vector, J_matrix


def initialize_spin_vector(h_vector, J_matrix):
    vertex = h_vector.shape[0]
    spin_vector = np.sign(h_vector).reshape(-1, 1)
    spin_vector[spin_vector == 0] = 1
    max_iterations = 10
    for _ in range(max_iterations):
        new_spin_vector = np.sign(h_vector + J_matrix @ spin_vector).reshape(-1, 1)
        new_spin_vector[new_spin_vector == 0] = 1
        if np.array_equal(spin_vector, new_spin_vector):
            break
        spin_vector = new_spin_vector
    return spin_vector


def find_energy_barrier(state_A, state_B, J, h):
    current_state = np.array(state_A).reshape(-1, 1)
    target_state = np.array(state_B).reshape(-1, 1)
    max_energy = energy_calculate(h, J, current_state, "no")
    while not np.array_equal(current_state, target_state):
        diff_indices = np.where(current_state != target_state)[0]
        flip_index = np.random.choice(diff_indices)
        current_state[flip_index] *= -1
        current_energy = energy_calculate(h, J, current_state, "no")
        max_energy = max(max_energy, current_energy)
    return max_energy


def compute_barriers(local_minima, J, h):
    local_minima = list(local_minima)
    num_minima = len(local_minima)
    barriers = np.zeros((num_minima, num_minima))
    for i in range(num_minima):
        for j in range(i + 1, num_minima):
            barrier = find_energy_barrier(local_minima[i], local_minima[j], J, h)
            barriers[i, j] = barrier
            barriers[j, i] = barrier
    return barriers


def calc_flip_counts(spin_history, tau):
    spins = np.hstack(spin_history)
    step_flips = np.sum(spins[:, 1:] != spins[:, :-1], axis=0)
    n_cycle = int(np.ceil(len(step_flips) / tau))
    flips_per_cycle = np.add.reduceat(step_flips, np.arange(0, len(step_flips), tau))[:n_cycle]
    return step_flips, flips_per_cycle

###############################################################
# ---------------- Simulation driver function  -------------- #
###############################################################

def run_simulation(graph_type, size, scale, cycle, algorithm, mean_range,
                   stall_prop, scale_mode):
    matrix_path = f"./matrix/{graph_type}/{graph_type}_N{size}_S{scale}_{scale_mode}_matrix.csv"
    feature_data = pd.read_csv(
        f"./matrix/{graph_type}/{graph_type}_N{size}_S{scale}_{scale_mode}_feature.csv",
        header=None,
    )
    vertex = int(feature_data.iloc[1, 1])
    min_energy_ideal = feature_data.iloc[4, 1]

    print("vertex : ", vertex, " min energy : ", min_energy_ideal)
    h_vector, J_matrix = get_graph_library(vertex, matrix_path)

    if median_scale == "yes":
        J_abs = np.sum(np.abs(J_matrix), axis=1)
        S = np.median(J_abs)
        h_vector = h_vector / S
        J_matrix = J_matrix / S
    else:
        S = 1
    print('parameter_scale = ', S)

    if algorithm in (0, 2, 3):
        sigma = np.mean([np.sqrt((vertex - 1) * np.var(J_matrix[j])) for j in range(vertex)])
        I0_min, I0_max = gamma / sigma, delta / sigma
        nrnd = np.float32(1)
    elif algorithm in (4, 5):
        sigma = np.mean([np.sqrt((vertex - 1) * np.var(J_matrix[j])) for j in range(vertex)])
        I0_min, I0_max = gamma / sigma, delta / sigma
        nrnd = np.mean(np.float32(0.67448975 / sigma * np.ones((vertex, 1))))
    else:  # SSA
        mean_each = []
        std_each = []
        for j in range(vertex):
            mean_each.append((vertex - 1) * np.mean(J_matrix[j]))
            std_each.append(np.sqrt((vertex - 1) * np.var(np.concatenate([J_matrix[j], -J_matrix[j]]))))
        sigma = np.mean(std_each)
        I0_min = np.float32(np.max(std_each) * 0.01 + np.min(np.abs(mean_each)))
        I0_max = np.float32(np.max(std_each) * 2 + np.min(np.abs(mean_each)))
        nrnd = np.mean(np.float32(0.67448975 * sigma * np.ones((vertex, 1))))

    beta = (I0_min / I0_max) ** (tau / (cycle - 1))
    print("nrnd :", nrnd)
    print(f'I0_min: {I0_min}, I0_max: {I0_max}, Control: {control}')

    energy_sum, energy_list = 0, []
    local_minima = set()

    for k in range(trial):
        ini_spin_vector = np.random.choice([-1, 1], size=(vertex, 1))
        Itanh_ini = (np.random.randint(0, 3, (vertex, 1)) - 1) * I0_min

        last_spin_vector, spin_history, energy_history, I0_history = annealing(
            vertex, tau, I0_min, I0_max, beta, nrnd, h_vector, J_matrix,
            ini_spin_vector, Itanh_ini, rand_type, cycle, control, algorithm,
            mean_range, stall_prop,
        )

        step_flips, flips_per_cycle = calc_flip_counts(spin_history, tau)
        print("各ステップで反転したスピン数:", step_flips[:10]/vertex)
        print("各サイクルで反転したスピン数:", flips_per_cycle[:10]/vertex)

        local_minima.add(tuple(last_spin_vector.flatten().tolist()))
        energy_val = energy_calculate(h_vector, J_matrix, last_spin_vector, "no")
        energy_sum += energy_val
        energy_list.append(energy_val)
        print(f"Trial {k + 1}, Energy value: {energy_val}, Optimal Value: {min_energy_ideal}")

    if verbose_plot.lower() == "yes":
        save_results(spin_history, energy_history)
        plot_results(energy_history, spin_history, I0_history, node_index)
        plt.plot(flips_per_cycle/vertex)
        plt.xlabel("Cycle index")
        plt.ylabel("#Flipped spins")
        plt.title("Simultaneous spin flips per cycle")
        plt.grid(True)
        plt.show()

    energy_avg = energy_sum / trial
    print("Min energy ideal:", float(min_energy_ideal) / S)
    print("Min annealing energy:", np.min(energy_list))
    print("Avg annealing energy:", energy_avg)
    print("Avg annealing energy%:", energy_avg / float(min_energy_ideal) * S)

    if algorithm == 0:
        alg_name = 'pSA'
    elif algorithm == 1:
        alg_name = 'SSA'
    elif algorithm == 2:
        alg_name = 'TApSA'
    elif algorithm == 3:
        alg_name = 'SpSA'

    data = [
        alg_name, graph_type, vertex, scale, scale_mode, cycle, tau, trial, min_energy_ideal,
        energy_avg, np.min(energy_list), np.max(energy_list),
        energy_avg / float(min_energy_ideal) * S, 
        np.mean(flips_per_cycle)/vertex, np.min(flips_per_cycle)/vertex, np.max(flips_per_cycle)/vertex,
        gamma, delta, rand_type,
        mean_range, stall_prop,
    ]
    save_summary(data)
    print("Results saved.")

###############################################################
# ----------------------- PRESET CONFIG --------------------- #
###############################################################

_PRESETS = {
    "quick": dict(
        cycle_list=[100],
        algorithm_list=[1],
        size_list=[100],
        type_list=["HNN"],
        scale_list=[1.0],
        scale_mode_list=["fixed"],
        mean_range_list=[5],
        stall_prop_list=[0.7],
        trial=1,
        tau=1,
        gamma=0.1,
        delta=10,
        rand_type=0,
        control=0,
        recool_cycle=1,
        median_scale="no",
        node_index=0,
        verbose_ann="no",
        verbose_plot="yes",
    ),
    "normal": dict(
        cycle_list=[1000],
        algorithm_list=[0, 1, 2, 3],
        size_list=[100, 250, 500, 1000, 2500],
        type_list=["HAA", "HZA", "HZN", "HNN", "HPN"],
        scale_list=[1.0],
        scale_mode_list=["fixed", "uniform", "normal"],
        mean_range_list=[5],
        stall_prop_list=[0.7],
        trial=10,
        tau=1,
        gamma=0.1,
        delta=10,
        rand_type=0,
        control=0,
        recool_cycle=1,
        median_scale="no",
        node_index=0,
        verbose_ann="no",
        verbose_plot="no",
    ),
    "TApSA": dict(
        cycle_list=[1000],
        algorithm_list=[2],
        size_list=[100, 250, 500, 1000, 2500],
        type_list=["HAA", "HZA", "HZN", "HNN", "HPN"],
        scale_list=[0.5, 1.0, 2.0],
        scale_mode_list=["fixed", "uniform", "normal"],
        mean_range_list=[1,3,5,7,9,11],
        stall_prop_list=[0.7],
        trial=10,
        tau=1,
        gamma=0.1,
        delta=10,
        rand_type=0,
        control=0,
        recool_cycle=1,
        median_scale="no",
        node_index=0,
        verbose_ann="no",
        verbose_plot="no",
    ),
    # Add future presets here, e.g. "large", "debug", ...
}


def preset_parameters(name: str):
    """Return parameter dictionary for given preset name."""
    try:
        return _PRESETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {', '.join(_PRESETS)}"
        ) from exc

###############################################################
# ---------------------- CLI utilities ---------------------- #
###############################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description="Annealing simulation driver")
    parser.add_argument(
        "--preset", choices=list(_PRESETS.keys()), default="normal",
        help="Preset name to use (default: normal)"
    )
    # convenience flags (still work):
    parser.add_argument("--quick", action="store_true", help="Alias for --preset quick")
    parser.add_argument("--normal", action="store_true", help="Alias for --preset normal")

    parser.add_argument("--trial", type=int, help="Override number of trials")
    parser.add_argument("--cycle", type=int, help="Override number of cycles per run")
    return parser.parse_args()

###############################################################
# ----------------------------- main ------------------------ #
###############################################################

def main():
    global trial, tau, gamma, delta, rand_type, control, recool_cycle
    global median_scale, node_index, verbose_ann, verbose_plot

    args = parse_arguments()
    # resolve preset name with backward‑compatibility flags
    preset_name = (
        "quick" if args.quick else "normal" if args.normal else args.preset
    )
    params = preset_parameters(preset_name)

    # allow simple overrides
    if args.trial is not None:
        params["trial"] = args.trial
    if args.cycle is not None:
        params["cycle_list"] = [args.cycle]

    # expose values to global scope expected by other functions
    trial = params["trial"]
    tau = params["tau"]
    gamma = params["gamma"]
    delta = params["delta"]
    rand_type = params["rand_type"]
    control = params["control"]
    recool_cycle = params["recool_cycle"]
    median_scale = params["median_scale"]
    node_index = params["node_index"]
    verbose_ann = params["verbose_ann"]
    verbose_plot = params["verbose_plot"]

    param_combinations = itertools.product(
        params["type_list"],
        params["size_list"],
        params["scale_list"],
        params["scale_mode_list"],
        params["cycle_list"],
        params["algorithm_list"],
        params["mean_range_list"],
        params["stall_prop_list"],
    )

    for graph_type, size, scale, scale_mode, cycle, algorithm, mean_range, stall_prop in param_combinations:
        print(
            f"Running simulation with graph_type={graph_type}, size={size}, scale={scale}, "
            f"scale_mode={scale_mode}, cycle={cycle}, algorithm={algorithm}, "
            f"mean_range={mean_range}, stall_prop={stall_prop}"
        )
        run_simulation(graph_type, size, scale, cycle, algorithm, mean_range, stall_prop, scale_mode)


if __name__ == "__main__":
    main()
