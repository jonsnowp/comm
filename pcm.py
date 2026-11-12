
import numpy as np
import matplotlib.pyplot as plt

fs = 1000
t = np.arange(0, 1, 1/fs)
f = 5
A = 2
offset = 2

x = A * np.sin(2 * np.pi * f * t) + offset

L_values = [4, 8, 16, 32, 64, 128]
sqnr_db_list = []
error_waveform = None
error_L = 16
x_q_list = []

for i, L in enumerate(L_values):
    b = int(np.log2(L))
    v_max, v_min = np.max(x), np.min(x)
    delta = (v_max - v_min) / L

    indices = np.floor((x - v_min) / delta)
    indices = np.clip(indices, 0, L - 1)

    x_q = v_min + (indices * delta) + (delta / 2)
    x_q_list.append(x_q)

    q_error = x - x_q

    P_signal = np.mean(x ** 2)
    P_noise = np.mean(q_error ** 2)

    sqnr = 10 * np.log10(P_signal / P_noise)
    sqnr_db_list.append(sqnr)

    if L == error_L:
        error_waveform = q_error

fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
fig1.canvas.manager.set_window_title('Experiment 1: PCM (Waveforms)')

axs1_flat = axs1.flatten()

for i, L in enumerate(L_values):
    b = int(np.log2(L))
    ax = axs1_flat[i]

    ax.plot(t, x, label='Original', color='black', alpha=0.6)
    ax.step(t, x_q_list[i], label=f'Quantized (b={b})',
            where='mid', color='red')

    ax.set_title(f'PCM: L={L}, SQNR={sqnr_db_list[i]:.2f} dB')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)

fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
fig2.canvas.manager.set_window_title('Experiment 1: PCM (Performance)')

axs2[0].plot(t, error_waveform, color='red')
axs2[0].set_title(f'Quantization Error Waveform (L={error_L})')
axs2[0].set_xlabel('Time')
axs2[0].set_ylabel('Error Amplitude')
axs2[0].grid(True)

bits_per_sample = np.log2(L_values)

axs2[1].plot(bits_per_sample, sqnr_db_list,
             marker='o', linestyle='-', color='blue')
axs2[1].set_title('SQNR vs Number of Bits per Sample')
axs2[1].set_xlabel('Bits per sample (b)')
axs2[1].set_ylabel('SQNR (dB)')
axs2[1].grid(True)

plt.show()
