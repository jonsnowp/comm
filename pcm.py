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

























"""
Aim: Generate a sinusoid, sample it, quantize it, binary encode it, and plot SQNR vs. bits per sample.
1) Generate the analog signal
Use a very high time resolution to mimic a continuous signal.
Logic and code, choose one: x(t) = A·sin(2πft) + offset
Square: x = A*np.sign(np.sin(2*np.pi*f*t)) + offset
Triangular: x = A*(2/np.pi)*np.arcsin(np.sin(2*np.pi*f*t)) + offset
Two sinusoids: x = A*np.sin(2*np.pi*f*t) + (A/2)*np.sin(2*np.pi*3*f*t) + offset
Code: t = np.arange(0, 1, 1/fs), x = A*np.sin(2*np.pi*f*t) + offset
Parameters: A = 2, f = 5 Hz, offset = 2, fs = 1000 Hz. The offset keeps the signal positive (0 to 4 V).
2) Sample the analog signal
Sample at a rate fs ≥ 2f (Nyquist). Here fs = 1000 Hz ≫ 10 Hz.
Logic: taking t at spacing 1/fs is the sampling step, so x holds the samples.
3) Quantize the samples
Number of levels L = 2^b, with L_values = [4, 8, 16, 32, 64, 128].
Find the range: v_max = max(x), v_min = min(x).
Step size: delta = (v_max - v_min) / L
Level index: indices = floor((x - v_min) / delta), then clip(indices, 0, L-1) so the maximum value does not overflow.
Quantized value (mid-point reconstruction): x_q = v_min + indices*delta + delta/2
Quantization error: q_error = x - x_q
4) Encode each quantized level into binary
Each level index is converted into a b-bit binary word, where b = log2(L).
Logic: np.binary_repr(index, width=b)
5) Generate the PCM bit stream
Concatenate all the binary words in time order into one serial stream.
6) Decode the received bits
Split the stream into groups of b bits and convert each back to an integer index (int(word, 2)).
7) Reconstruct the signal
Map the index back to a voltage: x_rec = v_min + index*delta + delta/2
This equals x_q (no channel errors assumed).
8) Evaluate performance
Mean square error: P_noise = np.mean(q_error**2)
Signal power: P_signal = np.mean(x**2)
SQNR(dB) = 10·log10(P_signal / P_noise)
Repeat for every L and store the results in sqnr_db_list.
Plot: x-axis = np.log2(L_values) (bits), y-axis = SQNR (dB), and also plot the original vs. quantized waveforms and the error waveform.
Expected result (to write in your conclusion)
SQNR increases as bits per sample increase.
Each extra bit adds about 6 dB (the theoretical formula is SQNR ≈ 6.02b + 1.76 dB for a full-scale sine).
The plot of SQNR vs. b is therefore approximately a straight line.
Quantization error stays within ±Δ/2, so it shrinks as L increases.
"""
