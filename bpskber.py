"""
Q1 / Q6: BPSK over AWGN -> ideal constellation + BER vs SNR (Eb/N0)

PROCEDURE
 1 Generate random bits            bits = np.random.randint(0, 2, N_bits)
 2 Map to BPSK symbols             0 -> +sqrt(Eb), 1 -> -sqrt(Eb)
 3 Plot ideal constellation        (I = +-sqrt(Eb), Q = 0), no noise
 4 For each SNR: add AWGN          N0 = Eb/snr_lin ; sigma = sqrt(N0/2)
 5 Detect                          real(y) >= 0 -> 0 else 1
 6 Simulated BER                   errors / N_bits
 7 Theoretical BER                 Pb = 0.5*erfc(sqrt(Eb/N0)) = Q(sqrt(2Eb/N0))
 8 Plot BER vs SNR (semilogy)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ---- CHANGE HERE (variations) ----
N_bits = 100000                       # more bits -> smoother curve at high SNR
Eb = 1
snr_db_range = np.arange(-2, 11)      # e.g. np.arange(0, 11, 2) or [0, 2, 4, 6, 8, 10]
sample_snr_db = 5                     # SNR for the received-constellation plot
SHOW_RECEIVED = True                  # False -> only ideal constellation + BER (Q1 exact)

bits = np.random.randint(0, 2, N_bits)                    # STEP 1
s = np.where(bits == 0, np.sqrt(Eb), -np.sqrt(Eb))        # STEP 2 (real symbols)

n_plots = 3 if SHOW_RECEIVED else 2
fig, axs = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

# STEP 3: ideal constellation (no noise), Q = 0 for BPSK
axs[0].scatter(s, np.zeros_like(s), color='blue', s=100, label='Ideal BPSK')
axs[0].set_title('Ideal BPSK Constellation'); axs[0].set_xlabel('In-Phase (I)'); axs[0].set_ylabel('Quadrature (Q)')
axs[0].axhline(0, color='k', lw=0.5); axs[0].axvline(0, color='k', lw=0.5)
axs[0].set_xlim(-2, 2); axs[0].set_ylim(-2, 2); axs[0].grid(True); axs[0].legend()

sim_ber, th_ber, sample_y = [], [], None
for snr_db in snr_db_range:
    snr_lin = 10**(snr_db/10)
    N0 = Eb/snr_lin
    sigma = np.sqrt(N0/2)                                  # STEP 4
    y = s + sigma*(np.random.randn(N_bits) + 1j*np.random.randn(N_bits))
    if snr_db == sample_snr_db:
        sample_y = y
    detected = np.where(np.real(y) >= 0, 0, 1)             # STEP 5
    sim_ber.append(np.sum(bits != detected)/N_bits)        # STEP 6
    th_ber.append(0.5*erfc(np.sqrt(Eb/N0)))                # STEP 7
    print(f'SNR={snr_db:3d} dB  sim BER={sim_ber[-1]:.5f}  theory={th_ber[-1]:.5f}')

k = 1
if SHOW_RECEIVED:
    axs[1].scatter(sample_y.real, sample_y.imag, s=1, color='red', alpha=0.5, label=f'Received ({sample_snr_db} dB)')
    axs[1].scatter([np.sqrt(Eb), -np.sqrt(Eb)], [0, 0], color='blue', marker='x', s=100, label='Ideal')
    axs[1].set_title('Received Constellation'); axs[1].set_xlabel('I'); axs[1].set_ylabel('Q')
    axs[1].grid(True); axs[1].legend(); k = 2

# STEP 8
axs[k].semilogy(snr_db_range, sim_ber, 'bo-', label='Simulated')
axs[k].semilogy(snr_db_range, th_ber, 'r-', lw=2, label='Theoretical')
axs[k].set_title('BPSK BER vs SNR (AWGN)'); axs[k].set_xlabel('Eb/N0 (dB)'); axs[k].set_ylabel('BER')
axs[k].grid(True, which='both', ls='--'); axs[k].legend()
plt.tight_layout(); plt.show()

# OTHER VARIATIONS:
#  * "SNR" given as Es/N0 : for BPSK Es = Eb, so same code.
#  * Plot BER with a different SNR list: only change snr_db_range.
#  * Expected: BER ~ 0.5 at very low SNR, ~1e-4 near 8 dB, falls rapidly.
