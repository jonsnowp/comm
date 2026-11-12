import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# STEP 1: GENERATE THE ANALOG SIGNAL
# =====================================================================
# CHANGE (different SAMPLING RATE / aliasing question):
#   Only change fs. Run once with fs >= 2f (e.g. 1000) and once with
#   fs < 2f (e.g. fs = 8 for f = 5) and compare the plots.
fs = 1000                 # sampling frequency (Hz)
f = 5                     # signal frequency (Hz)
A = 2                     # amplitude
offset = 2                # DC offset (set 0 for a bipolar signal -A..+A)
t = np.arange(0, 1, 1/fs)

# CHANGE (different SIGNAL): keep ONE line below, comment out the others.
x = A * np.sin(2 * np.pi * f * t) + offset                    # sine (original)
# x = A * np.sign(np.sin(2 * np.pi * f * t)) + offset         # square wave
# x = A * (2/np.pi) * np.arcsin(np.sin(2*np.pi*f*t)) + offset # triangular wave
# x = A*np.sin(2*np.pi*f*t) + (A/2)*np.sin(2*np.pi*3*f*t) + offset  # sum of two sinusoids

# =====================================================================
# STEP 2: SAMPLE THE ANALOG SIGNAL
# =====================================================================
# Sampling happens through t = np.arange(0, 1, 1/fs); x holds the samples.
# Nyquist condition: fs >= 2 * f_max  (f_max = highest frequency in x)

# =====================================================================
# SETTINGS FOR STEPS 3 TO 8
# =====================================================================
# CHANGE (different thing to PLOT):
#   SQNR vs bits  -> L_values = [4, 8, 16, 32, 64, 128]
#   Single quantizer with L = 8 -> L_values = [8]
L_values = [4, 8, 16, 32, 64, 128]

# CHANGE (non-uniform quantization): True for mu-law companding
USE_MU_LAW = False
mu = 255

# CHANGE (channel errors): probability of a bit flip. 0 = no errors.
# e.g. BER = 0.01 for 1% bit errors
BER = 0.0

sqnr_db_list = []
x_rec_list = []
error_L = L_values[min(2, len(L_values) - 1)]   # which L to show error for
error_waveform = None

for L in L_values:
    b = int(np.log2(L))                 # bits per sample

    # -----------------------------------------------------------------
    # STEP 3: QUANTIZE THE SAMPLES
    # -----------------------------------------------------------------
    # CHANGE (mu-law): compress BEFORE quantizing
    if USE_MU_LAW:
        x_max = np.max(np.abs(x))
        x_norm = x / x_max                                   # normalize to +-1
        y = np.sign(x_norm) * np.log(1 + mu * np.abs(x_norm)) / np.log(1 + mu)
    else:
        y = x                                                # uniform: no compression

    v_max, v_min = np.max(y), np.min(y)
    delta = (v_max - v_min) / L                              # step size

    indices = np.floor((y - v_min) / delta)                  # level index
    indices = np.clip(indices, 0, L - 1).astype(int)         # keep within 0..L-1

    # -----------------------------------------------------------------
    # STEP 4: ENCODE EACH QUANTIZED LEVEL INTO BINARY
    # -----------------------------------------------------------------
    codewords = [np.binary_repr(i, width=b) for i in indices]

    # -----------------------------------------------------------------
    # STEP 5: GENERATE THE PCM BIT STREAM
    # -----------------------------------------------------------------
    bitstream = ''.join(codewords)

    # -----------------------------------------------------------------
    # (CHANNEL) CHANGE: add bit errors between step 5 and step 6
    # -----------------------------------------------------------------
    if BER > 0:
        bits = np.array(list(bitstream), dtype=int)
        flips = (np.random.rand(len(bits)) < BER).astype(int)
        bits = bits ^ flips                                  # flip selected bits
        bitstream = ''.join(map(str, bits))

    # -----------------------------------------------------------------
    # STEP 6: DECODE THE RECEIVED BITS
    # -----------------------------------------------------------------
    decoded = np.array([int(bitstream[k:k+b], 2)
                        for k in range(0, len(bitstream), b)])

    # -----------------------------------------------------------------
    # STEP 7: RECONSTRUCT THE SIGNAL
    # -----------------------------------------------------------------
    y_rec = v_min + decoded * delta + delta / 2              # mid-point value

    # CHANGE (mu-law): expand AFTER decoding
    if USE_MU_LAW:
        x_rec = np.sign(y_rec) * ((1 + mu) ** np.abs(y_rec) - 1) / mu * x_max
    else:
        x_rec = y_rec
    x_rec_list.append(x_rec)

    # -----------------------------------------------------------------
    # STEP 8: EVALUATE PERFORMANCE (MSE and SQNR)
    # -----------------------------------------------------------------
    q_error = x - x_rec
    P_signal = np.mean(x ** 2)
    P_noise = np.mean(q_error ** 2)                          # mean square error
    sqnr = 10 * np.log10(P_signal / P_noise)
    sqnr_db_list.append(sqnr)

    if L == error_L:
        error_waveform = q_error

# =====================================================================
# PLOTS
# =====================================================================
n = len(L_values)
ncols = min(3, n)
nrows = int(np.ceil(n / ncols))
fig1, axs1 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                          constrained_layout=True, squeeze=False)
axs1_flat = axs1.flatten()

for i, L in enumerate(L_values):
    b = int(np.log2(L))
    ax = axs1_flat[i]
    ax.plot(t, x, label='Original', color='black', alpha=0.6)
    ax.step(t, x_rec_list[i], label=f'Quantized (b={b})', where='mid', color='red')
    ax.set_title(f'PCM: L={L}, SQNR={sqnr_db_list[i]:.2f} dB')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
for j in range(n, len(axs1_flat)):          # hide unused subplots
    axs1_flat[j].axis('off')

fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

axs2[0].plot(t, error_waveform, color='red')
axs2[0].set_title(f'Quantization Error Waveform (L={error_L})')
axs2[0].set_xlabel('Time')
axs2[0].set_ylabel('Error Amplitude')
axs2[0].grid(True)

bits_per_sample = np.log2(L_values)
axs2[1].plot(bits_per_sample, sqnr_db_list, marker='o', linestyle='-', color='blue')
axs2[1].set_title('SQNR vs Number of Bits per Sample')
axs2[1].set_xlabel('Bits per sample (b)')
axs2[1].set_ylabel('SQNR (dB)')
axs2[1].grid(True)

plt.show()
