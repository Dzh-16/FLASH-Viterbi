from Viterbi import Sieve
from sieve_beam_search import SIEVE_BEAMSEARCH
import numpy as np
import random
import time
import sys
import pickle
import os

# --- PATCH: helpers for size accounting (shallow + numpy buffer) ---
import sys
def _size_np_or_obj(x):
    return sys.getsizeof(x) + (getattr(x, "nbytes", 0) if hasattr(x, "nbytes") else 0)

def _size_path(path):
    if isinstance(path, list):
        return sys.getsizeof(path) + sum(_size_np_or_obj(p) for p in path)
    return _size_np_or_obj(path)

def create_A_b(n_nodes=100, sd=1, prob=0.2):
    np.random.seed(sd)
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    allstates = [x for x in range(n_nodes)]

    for state in range(n_nodes):
        edge_per_node = np.random.binomial(n_nodes, p=prob, size=None)
        state_connections = np.random.choice(allstates, size=edge_per_node, replace=False)
        ps = np.random.uniform(0.01, 1, size=edge_per_node)

        for i in range(edge_per_node):
            connection = state_connections[i]
            p = ps[i]
            matrix[state][connection] = p

    # Normalize matrix
    for i in range(n_nodes):
        s = np.sum(matrix[i,])
        matrix[i,] = matrix[i,] / np.sum(matrix[i,])

    return matrix


def create_B(n_observables=100, n_states=100, sd=1):
    ''' Create matrix of uniform emission probabilities '''
    np.random.seed(sd)

    B = np.random.uniform(0.1, 1, (n_states, n_observables))
    B = B / B.sum(axis=1)[:, None]

    return B


if __name__ == '__main__':

    # Seed for data generation
    sd = 12
    # Number of observable symbols
    n_ob = 50
    # Number of states
    K = 128
    random.seed(sd)
    # Vector of observations
    T = 50
    beam_width = 128
    prob = 0.253

    file = open(f'ANS_K{K}_T{T}_prob{prob}_beam_width{beam_width}.txt', 'w')
    file.write(f"sd={sd}, n_ob={n_ob}, K={K}, T={T}, beam_width={beam_width}, prob={prob}\n")

    y = [random.randint(0, n_ob - 1) for _ in range(T)]

    # Generate simple data
    A = create_A_b(n_nodes=K, sd=sd, prob=prob)
    B = create_B(n_observables=n_ob, n_states=K, sd=sd)

    # Uniform initial probabilities
    pi = np.full(K, 1 / K)

    # Save A, B, Pi, y to txt files with float format
    np.savetxt(f'A_K{K}_T{T}_prob{prob}.txt', A, fmt='%.16f')
    np.savetxt(f'B_K{K}_T{T}_prob{prob}.txt', B, fmt='%.16f')
    np.savetxt(f'Pi_K{K}_T{T}_prob{prob}.txt', pi, fmt='%.16f', newline=' ')
    np.savetxt(f'ob_K{K}_T{T}_prob{prob}.txt', y, fmt='%d', newline=' ')

    print("Starting Vanilla Viterbi .. \n")
    vit = Sieve(pi, A, B, y)
    start_time = time.time()
    vit.viterbi()
    time_taken = round(time.time() - start_time, 5)
    print(f"path:", vit.path)
    print("Vanilla Viterbi done")
    print(f"Time: {time_taken}s\n")
    file.write(f"Vanilla Viterbi Time: {time_taken}s\n")
    file.write(
        "Mem(nonPath+PthSize):"
        f"{getattr(vit, 'memory_bytes', 0)}"
        f"+{_size_np_or_obj(getattr(vit, 'path', []))}\n"
    )

    print("Starting Checkpoint Viterbi .. \n")
    vit = Sieve(pi, A, B, y)
    start_time = time.time()
    x_checkpoint = vit.viterbi_checkpoint()
    time_taken = round(time.time() - start_time, 5)
    print("Checkpoint Viterbi done .. \n" + "path: ", list(map(int, x_checkpoint)))
    print(f"Time: {time_taken}s\n")
    file.write(f"Checkpoint Viterbi Time: {time_taken}s\n")
    file.write(
        "Mem(nonPath+PthSize):"
        f"{getattr(vit, 'memory_bytes', 0)}"
        f"+{_size_np_or_obj(getattr(vit, 'x_checkpoint', []))}\n"
    )

    print("Starting Sieve Middle Path.. \n" + "path edges: ")
    vit = Sieve(pi, A, B, y)
    indices = [x for x in range(K)]
    pi = np.full(K, 1 / K)
    vit.initial_state = None
    start_time = time.time()
    vit.sieve_middlepath(indices, A, B, y, Pi=pi, K=K)
    time_taken = round(time.time() - start_time, 5)
    vit.pretty_print_path(vit.mp_path)
    print("Sieve Middlepath done ..")
    print(f"Time: {time_taken}s\n")
    file.write(f"Sieve Middlepath Time: {time_taken}s\n")
    file.write(
        "Mem(nonPath[nonBFS/withBFS]+PthSize):"
        f"[{vit.memory_bytes},{vit.memory_bytes2}]"
        f"+{_size_np_or_obj(getattr(vit, 'mp_path', []))}\n"
    )


    # Save preprocessed data with parameterized file name
    data_file = f'preprocessed_data_K{K}_T{T}_prob{prob}_beam_width{beam_width}.pkl'

    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            pi = data['pi']
            A_out = data['A_out']
            A_in = data['A_in']
            acustic_costs = data['acustic_costs']
        print("Preprocessed data loaded successfully.")
    else:
        print("Preprocessed data not found, starting preprocessing...")
        A_in = [[] for _ in range(K)]
        A_out = [[] for _ in range(K)]
        acustic_costs = [{} for _ in range(n_ob)]
        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i][j] != 0:
                    A_in[j].append((i, np.log(A[i][j])))
                    A_out[i].append((j, np.log(A[i][j])))
        for i in range(len(B)):
            for step in range(len(B[i])):
                if B[i][step] != 0:
                    tmp_tmp_B = np.log(B[i][step])
                    for j in range(len(B)):
                        acustic_costs[step][(j, i)] = tmp_tmp_B
        pi = np.full(K, np.log(1 / K))

        # Save preprocessed data to file
        with open(data_file, 'wb') as f:
            pickle.dump({
                'pi': pi,
                'A_out': A_out,
                'A_in': A_in,
                'acustic_costs': acustic_costs
            }, f)
        print("Preprocessing completed and saved successfully.")

    print("Starting SIEVE_BEAMSEARCH\n" + "path edges: ")
    indices = [x for x in range(K)]
    bs = SIEVE_BEAMSEARCH(pi, A_out, A_in, acustic_costs, beam_width)
    descendants_pruning_root_memory = bs.viterbi_preprocessing_descendants_pruning_root(indices, T, K)
    ancestors_pruning_root_memory = bs.viterbi_preprocessing_ancestors_pruning_root(indices, T, K)
    # print(f"descendants_pruning_root_memory: {descendants_pruning_root_memory}")
    # print(f"ancestors_pruning_root_memory: {ancestors_pruning_root_memory}")
    start_time = time.time()
    bs.viterbi_space_efficient(indices, frames=y, Pi=pi, K=K)
    time_taken = round(time.time() - start_time, 5)
    bs.pretty_print_path(bs.path)
    print("SIEVE_BEAMSEARCH done ..")
    print(f"Time: {time_taken}s\n")
    file.write(f"SIEVE_BEAMSEARCH Time: {time_taken}s\n")
    file.write(
        f"Mem(nonPath[nonBFS/withBFS]+PthSize):[{bs.memory_bytes},{bs.memory_bytes2}]+{_size_path(bs.path)}\n"
    )


    print("Starting SIEVE_BEAMSEARCH middlepath\n" + "path edges: ")
    bs = SIEVE_BEAMSEARCH(pi, A_out, A_in, acustic_costs, beam_width)
    descendants_pruning_root_memory = bs.viterbi_preprocessing_descendants_pruning_root(indices, T, K)
    ancestors_pruning_root_memory = bs.viterbi_preprocessing_ancestors_pruning_root(indices, T, K)
    # print(f"descendants_pruning_root_memory: {descendants_pruning_root_memory}")
    # print(f"ancestors_pruning_root_memory: {ancestors_pruning_root_memory}")
    start_time = time.time()
    bs.viterbi_middlepath(indices, frames=y, Pi=pi, K=K)
    time_taken = round(time.time() - start_time, 5)
    bs.pretty_print_path(bs.path)
    print("SIEVE_BEAMSEARCH middlepath done ..")
    print(f"Time: {time_taken}s\n")
    file.write(f"SIEVE_BEAMSEARCH Middlepath Time: {time_taken}s\n")
    file.write(
        f"Mem(nonPath[nonBFS/withBFS]+PthSize):[{bs.memory_bytes},{bs.memory_bytes2}]+{_size_path(bs.path)}\n"
    )

    file.close()
