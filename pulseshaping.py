import numpy as np
import matplotlib.pyplot as plt

def generate_srrc(beta, N_sym, L):
    t = np.arange(-N_sym/2, N_sym/2 + 1/L, 1/L)
    p = np.zeros(len(t))
    if beta == 0:
        beta = 1e-8
    for i, tc in enumerate(t):
        if tc == 0:
            p[i] = 1 - beta + 4*beta/np.pi
        elif np.isclose(np.abs(tc), 1/(4*beta)):
            p[i] = (beta/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta))
                + (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
        else:
            num = (
                np.sin(np.pi*tc*(1-beta))
                + 4*beta*tc*np.cos(np.pi*tc*(1+beta))
            )
            den = np.pi*tc*(1-(4*beta*tc)**2)
            p[i] = num/den
    return p, t

N_bits = 50000
L = 4
N_sym = 8
default_beta = 0.4
snr_db_range = np.arange(0, 16, 2)
plot_limit = 20

bits = np.random.randint(0, 2, N_bits)
symbols = np.where(bits == 0, 1, -1)
upsampled = np.zeros(len(symbols)*L)
upsampled[::L] = symbols

p_default, t_pulse = generate_srrc(default_beta, N_sym, L)
s_tx = np.convolve(upsampled, p_default, mode='full')

snr_lin_test = 10**(8/10)
P_s = np.mean(s_tx**2)
noise_test = np.sqrt(P_s/snr_lin_test) * np.random.randn(len(s_tx))
r_rx = s_tx + noise_test

g = p_default[::-1]
y_mf = np.convolve(r_rx, g, mode='full')
delta = (len(p_default)-1)//2
total_delay = 2*delta
y_sampled = y_mf[total_delay::L][:N_bits]
detected_bits = np.where(y_sampled >= 0, 0, 1)

fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
fig1.canvas.manager.set_window_title('Experiment 4: Pulse Shaping (Signals)')
axs1_flat = axs1.flatten()

axs1_flat[0].stem(range(plot_limit), bits[:plot_limit], basefmt=" ")
axs1_flat[0].set_title('Generated Binary Message Sequence')

axs1_flat[1].stem(
    range(plot_limit*L), upsampled[:plot_limit*L], basefmt=" "
)
axs1_flat[1].set_title('Upsampled Baseband Signal')

axs1_flat[2].plot(t_pulse, p_default, color='green')
axs1_flat[2].set_title(f'RRC Pulse (alpha)={default_beta}')

axs1_flat[3].plot(s_tx[:plot_limit*L], color='purple')
axs1_flat[3].set_title('Transmitted Baseband Signal')

axs1_flat[4].plot(r_rx[:plot_limit*L], color='red')
axs1_flat[4].set_title('Received Signal (AWGN SNR = 8 dB)')

axs1_flat[5].plot(
    y_mf[:plot_limit*L + total_delay], color='orange'
)
axs1_flat[5].set_title('Matched Filter Output')

fig2, axs2 = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
fig2.canvas.manager.set_window_title('Experiment 4: Pulse Shaping (Performance)')
axs2_flat = axs2.flatten()

axs2_flat[0].stem(range(plot_limit), y_sampled[:plot_limit], basefmt=" ")
axs2_flat[0].set_title('Sampled Symbols (After Downsampling)')

axs2_flat[1].stem(
    range(plot_limit), detected_bits[:plot_limit], basefmt=" "
)
axs2_flat[1].set_title('Detected Bit Sequence')

betas_to_test = [0.0, 0.25, 0.5, 1.0]
for b_val in betas_to_test:
    p_temp, t_temp = generate_srrc(b_val, N_sym, L)
    axs2_flat[2].plot(t_temp, p_temp, label=f'(beta)={b_val}')
axs2_flat[2].set_title('RRC Pulse Comparison')
axs2_flat[2].legend()

for b_val in betas_to_test:
    p_temp, _ = generate_srrc(b_val, N_sym, L)
    s_temp = np.convolve(
        upsampled[:plot_limit*L], p_temp, mode='full'
    )
    axs2_flat[3].plot(
        s_temp[:plot_limit*L], alpha=0.7, label=f'(beta)={b_val}'
    )
axs2_flat[3].set_title('Transmitted Baseband Comparison')
axs2_flat[3].legend()

ber_vs_beta = []
test_snr_lin = 10**(6/10)
for b_val in betas_to_test:
    p_temp, _ = generate_srrc(b_val, N_sym, L)
    s_temp = np.convolve(upsampled, p_temp, mode='full')
    n_temp = np.sqrt(
        np.mean(s_temp**2)/test_snr_lin
    ) * np.random.randn(len(s_temp))
    y_mf_temp = np.convolve(
        s_temp + n_temp, p_temp[::-1], mode='full'
    )
    samp = y_mf_temp[(len(p_temp)-1)::L][:N_bits]
    ber_vs_beta.append(
        np.sum(bits != np.where(samp >= 0, 0, 1))/N_bits
    )

axs2_flat[4].plot(betas_to_test, ber_vs_beta, 'go-')
axs2_flat[4].set_title('BER vs Roll-off Factor (SNR = 6 dB)')
axs2_flat[4].set_xlabel('Roll-off (beta)')
axs2_flat[4].set_ylabel('BER')
axs2_flat[4].grid(True)

ber_sim = []
for snr in snr_db_range:
    n = np.sqrt(
        P_s/(10**(snr/10))
    ) * np.random.randn(len(s_tx))
    y_filt = np.convolve(s_tx + n, g, mode='full')
    samp = y_filt[total_delay::L][:N_bits]
    ber_sim.append(
        np.sum(bits != np.where(samp >= 0, 0, 1))/N_bits
    )

ber_theory = 0.5 * (
    1 - np.sqrt(
        (10**(snr_db_range/10)) /
        (1 + 10**(snr_db_range/10))
    )
)

axs2_flat[5].semilogy(snr_db_range, ber_sim, 'bo-', label='Simulated')
axs2_flat[5].semilogy(snr_db_range, ber_theory, 'r-', label='Theoretical')
axs2_flat[5].set_title('BER vs SNR')
axs2_flat[5].set_xlabel('SNR (dB)')
axs2_flat[5].set_ylabel('BER')
axs2_flat[5].legend()
axs2_flat[5].grid(True, which='both', ls='--')

plt.show()




























"""
1) Generate the bits and map them to symbols
Code: bits = np.random.randint(0, 2, N_bits), symbols = np.where(bits == 0, 1, -1)
Parameters: N_bits = 50000, oversampling factor L = 4, pulse span N_sym = 8, roll-off β = 0.4.
2) Upsample the symbols
Logic: insert L−1 zeros between symbols.
Code: upsampled = np.zeros(len(symbols)*L), upsampled[::L] = symbols
3) Generate the SRRC pulse
Time axis (in symbol periods): t = np.arange(-N_sym/2, N_sym/2 + 1/L, 1/L)
At t = 0: p = 1 − β + 4β/π
At |t| = 1/(4β): p = (β/√2)[(1 + 2/π) sin(π/4β) + (1 − 2/π) cos(π/4β)]
Otherwise: p = [sin(πt(1−β)) + 4βt cos(πt(1+β))] / [πt(1 − (4βt)²)]
If β = 0, use a tiny value (1e-8) to avoid division by zero.
4) Pulse-shape the signal (transmitter)
Logic: convolve the upsampled symbols with the pulse.
Code: s_tx = np.convolve(upsampled, p, mode='full')
5) Pass through the AWGN channel
Signal power: P_s = np.mean(s_tx**2)
Noise: noise = np.sqrt(P_s/snr_lin) * np.random.randn(len(s_tx)), and r_rx = s_tx + noise.
6) Matched filter (receiver)
Logic: the matched filter is the time-reversed pulse.
Code: g = p[::-1], y_mf = np.convolve(r_rx, g, mode='full')
7) Sample at the symbol instants
Delay of the two filters: total_delay = 2*delta, where delta = (len(p) - 1)//2
Code: y_sampled = y_mf[total_delay::L][:N_bits]
8) Detect the bits
detected_bits = np.where(y_sampled >= 0, 0, 1)
9) Evaluate performance
BER = np.sum(bits != detected_bits)/N_bits
Repeat for SNR = 0 to 14 dB, and for β = 0, 0.25, 0.5, 1.
Plot the bits, upsampled signal, SRRC pulse, transmitted signal, received signal, matched filter output, sampled symbols, detected bits, BER vs. β, and BER vs. SNR.
Expected result: The matched filter output is sampled at the symbol instants with no ISI, so BER falls with increasing SNR. A larger β gives a wider pulse in frequency and a shorter one in time
"""
