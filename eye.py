
import numpy as np
import matplotlib.pyplot as plt

def generate_srrc(alpha, N_sym, L):
    t = np.arange(-N_sym/2, N_sym/2 + 1/L, 1/L)
    p = np.zeros(len(t))

    if alpha == 0:
        alpha = 1e-8

    for i, tc in enumerate(t):
        if tc == 0:
            p[i] = 1 - alpha + 4*alpha/np.pi

        elif np.isclose(np.abs(tc), 1/(4*alpha)):
            p[i] = (alpha/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*alpha))
                + (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
            )

        else:
            p[i] = (
                np.sin(np.pi*tc*(1-alpha))
                + 4*alpha*tc*np.cos(np.pi*tc*(1+alpha))
            ) / (
                np.pi*tc*(1-(4*alpha*tc)**2)
            )

    return p


N_bits = 2000
L = 4
N_sym = 8

bits = np.random.randint(0, 2, N_bits)
symbols = np.where(bits == 0, 1, -1)

upsampled = np.zeros(len(symbols)*L)
upsampled[::L] = symbols

scenarios_snr = [(0.4, 0), (0.4, 4), (0.4, 8)]
scenarios_alpha = [(0.1, 4), (0.5, 4), (0.9, 4)]

all_scenarios = scenarios_snr + scenarios_alpha

fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
axs_flat = axs.flatten()

for i, (alpha, snr_db) in enumerate(all_scenarios):
    p = generate_srrc(alpha, N_sym, L)

    s = np.convolve(upsampled, p, mode='full')

    snr_lin = 10**(snr_db/10)

    noise = np.sqrt(
        np.mean(s**2)/snr_lin
    ) * np.random.randn(len(s))

    r = s + noise
    y = np.convolve(r, p[::-1], mode='full')

    start_idx = len(p)-1
    nSamples = 3*L
    nTraces = 100

    ax = axs_flat[i]

    for j in range(nTraces):
        idx = start_idx + j*L

        if idx + nSamples < len(y):
            ax.plot(
                range(nSamples),
                y[idx:idx+nSamples],
                'b-',
                alpha=0.3
            )

    if i < 3:
        ax.set_title(
            f'SNR Effect: (alpha)={alpha}, SNR={snr_db}dB'
        )
    else:
        ax.set_title(
            f'Roll-off Effect: (alpha)={alpha}, SNR={snr_db}dB'
        )

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

plt.show()


























"""
1) Generate the bits and map them to symbols
Code: bits = np.random.randint(0, 2, N_bits), symbols = np.where(bits == 0, 1, -1)
Parameters: N_bits = 2000, L = 4, N_sym = 8.
2) Upsample the symbols
upsampled[::L] = symbols
3) Generate the SRRC pulse
Use the same SRRC formula as Experiment 3, with roll-off α.
4) Pulse-shape the signal
s = np.convolve(upsampled, p, mode='full')
5) Add AWGN noise
noise = np.sqrt(np.mean(s**2)/snr_lin) * np.random.randn(len(s)), r = s + noise
6) Matched filter
y = np.convolve(r, p[::-1], mode='full')
7) Slice the output into overlapping traces
Start index: start_idx = len(p) - 1 (the filter delay).
Each trace has nSamples = 3*L points.
Take nTraces = 100 traces, shifting the start by L samples each time: y[idx : idx + nSamples]
8) Plot the eye diagram
Plot all traces on the same axes with alpha = 0.3 transparency, so overlapping lines build up the eye.
Run 6 scenarios:
Effect of SNR: α = 0.4 with SNR = 0, 4, 8 dB.
Effect of roll-off: SNR = 4 dB with α = 0.1, 0.5, 0.9.

Expected result:

Higher SNR gives a wider, clearer eye opening.
Lower SNR makes the traces spread out and the eye closes.
A small α gives a narrow eye that is sensitive to timing errors.
A large α gives a wider horizontal eye opening and less ISI at the sampling instant.
"""
