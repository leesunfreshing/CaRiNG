import os
import numpy as np
from scipy.stats import ortho_group

def tanh_activation(x):
    return np.tanh(x)

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)

def get_j(t):
    t = (t + 1) % 9
    if t < 3:
        return 0
    elif t >= 3 and t < 6:
        return 2
    else:
        return 1

def invertible_noise_function(z_prev, noise, alpha=0.1):
    return z_prev * (1 + alpha * noise)

def recover_noise(z_t, z_prev, alpha=0.1, epsilon=1e-8):
    return (z_t / (z_prev + epsilon) - 1) / alpha

def generate_data(seed, lags, length, latent_size, noise_scale, num_sequences, root_dir):
    np.random.seed(seed)

    # Create data path
    path = os.path.join(root_dir, f"seed{seed}_mixinglag{lags}")
    os.makedirs(path, exist_ok=True)

    # Generate transition matrices for each sequence
    transitions = []
    for _ in range(num_sequences):
        sequence_transitions = []
        while len(sequence_transitions) < lags:
            A = np.random.uniform(-1, 1, (latent_size, latent_size))
            for i in range(latent_size):
                A[:, i] /= np.linalg.norm(A[:, i])
            
            if np.linalg.cond(A) < 10:
                sequence_transitions.append(A)
        transitions.append(sequence_transitions)

    # Generate mixing matrices for each sequence
    mixing_matrices = []
    for _ in range(num_sequences):
        sequence_mixing = [ortho_group.rvs(latent_size) for _ in range(1)]  # 1 layer of mixing
        mixing_matrices.append(sequence_mixing)

    # Initialize latent variables
    z = np.random.normal(0, 1, (num_sequences, lags, latent_size))
    z = (z - np.mean(z, axis=0, keepdims=True)) / np.std(z, axis=0, keepdims=True)

    z_series = []
    x_series = []
    transition_no_j_series = []
    mixing_matrix_no_j_series = []

    for t in range(length):
        j = get_j(t)
        if t >= 1:
            j_prev = get_j(t-1)
        
        # Transition function
        z_t = np.zeros((num_sequences, latent_size))
        noise = np.random.normal(0, noise_scale, (num_sequences, latent_size))
        transition_no_j_t = []

        for seq in range(num_sequences):
            z_t_seq = z_t[seq]
            for l in range(lags):
                transition_no_j = transitions[seq][l].copy()
                transition_no_j[:, j] = 0  # Zero out j-th column given the right multiplier
                if t >= 1:
                    transition_no_j[j_prev, :] = 0  # Zero out j_prev-th row given the right multiplier
                z_t_seq += leaky_ReLU(np.dot(z[seq, l, :], transition_no_j), 0.2)
            
            # Apply non-additive noise
            z_t_seq = invertible_noise_function(z_t_seq, noise[seq])
            
            z_t[seq] = z_t_seq
            transition_no_j_t.append(transition_no_j)
            
        # Set z^j_t to have no parents (only noise)
        z_t[:, j] = np.random.uniform(-noise_scale, noise_scale, num_sequences)
        
        # Apply activation to all components
        z_t = leaky_ReLU(z_t, 0.2)
        
        # Mixing function (excluding j)
        x_t = z_t.copy()
        mixing_matrix_no_j_t = []
        
        for seq in range(num_sequences):
            x_t_seq = x_t[seq]
            seq_mixing_matrix_no_j = []
            for mixing_matrix in mixing_matrices[seq]:
                mixing_matrix_no_j = mixing_matrix.copy()
                mixing_matrix_no_j[j, :] = 0  # Zero out j-th row given the right multiplier
                x_t_seq = np.dot(x_t_seq, mixing_matrix_no_j)
                seq_mixing_matrix_no_j.append(mixing_matrix_no_j)
                
                x_t_seq = leaky_ReLU(x_t_seq, 0.2)
            
            x_t[seq] = x_t_seq
            mixing_matrix_no_j_t.append(seq_mixing_matrix_no_j)
        
        z_series.append(z_t)
        x_series.append(x_t)
        transition_no_j_series.append(transition_no_j_t)
        mixing_matrix_no_j_series.append(mixing_matrix_no_j_t)
        
        # Update z history
        z = np.roll(z, -1, axis=1)
        z[:, -1, :] = z_t

    z_array = np.array(z_series).transpose(1, 0, 2)
    x_array = np.array(x_series).transpose(1, 0, 2)

    # Save data
    np.savez(os.path.join(path, "data.npz"), yt=z_array, xt=x_array)

    # Save transition matrices
    np.save(os.path.join(path, "transitions.npy"), np.array(transitions))

    # Save mixing matrices
    np.save(os.path.join(path, "mixing_matrices.npy"), np.array(mixing_matrices))

    # Save transition_no_j sequences
    transition_no_j_array = np.array(transition_no_j_series)
    np.save(os.path.join(path, "transition_no_j_sequences.npy"), transition_no_j_array)

    # Save mixing_matrix_no_j sequences
    mixing_matrix_no_j_array = np.array(mixing_matrix_no_j_series)
    np.save(os.path.join(path, "mixing_matrix_no_j_sequences.npy"), mixing_matrix_no_j_array)

    return z_array, x_array, transition_no_j_array, mixing_matrix_no_j_array

# Example usage
seed = 770
lags = 1
length = 9  # 9 steps per sequence
latent_size = 3
noise_scale = 0.1
num_sequences = 40000  # 40,000 sequences
root_dir = "/home/lidwh/CaRiNG/datasets_missing"

z, x, transitions, mixing_matrices = generate_data(seed, lags, length, latent_size, noise_scale, num_sequences, root_dir)

print("z shape:", z.shape)
print("x shape:", x.shape)
print("transitions shape:", transitions.shape)
print("mixing_matrices shape:", mixing_matrices.shape)
print(f"Data saved in: {os.path.join(root_dir, f'seed{seed}_mixinglag{lags}')}")