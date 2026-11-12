"""
Q4 / Q10: QPSK over AWGN -> BER vs SNR

PROCEDURE
 1 Generate random bits, group into pairs          bit_pairs = bits.reshape(-1, 2)
 2 Map each pair to a QPSK symbol (Gray)           00->1+j, 01->-1+j, 11->-1-j, 10->1-j (times sqrt(Eb))
 3 For each Eb/N0: add complex AWGN                N0 = Eb/snr_lin ; sigma = sqrt(N0/2) on I and Q
 4 Detect by quadrant                              b0 = (imag<0), b1 = (real<0)
 5 Simulated BER                                   errors / N_bits
 6 Theoretical BER                                 Pb = 0.5*erfc(sqrt(Eb/N0))
 7 Plot constellation (ideal, received) and BER vs SNR
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ---- CHANGE HERE (variations) ----
N_bits = 200000                         # must be even
Eb = 1
snr_db_range = np.arange(-2, 11)        # e.g. np.arange(0, 11, 2)
sample_snr_db = 5
GRAY = True                             # False -> natural mapping 00,01,10,11 (BER becomes worse)

bits = np.random.randint(0, 2, N_bits)
b0, b1 = bits[0::2], bits[1::2]
N_sym = len(b0)

# STEP 2: vectorised mapping (same table as the loop version)
if GRAY:
    re = 1 - 2*b1          # b1 = 1 -> left half
    im = 1 - 2*b0          # b0 = 1 -> lower half
else:                      # natural: 00->1+j, 01->-1+j, 10->1-j, 11->-1-j  (b0 sets Q, b1 sets I; same as above)
    # CHANGE: for a different natural mapping use I from (b0, b1) as a 2-bit number
    re = 1 - 2*(b0 ^ b1)
    im = 1 - 2*b0
s = np.sqrt(Eb)*(re + 1j*im)

sim_ber, th_ber, sample_y = [], [], None
for snr_db in snr_db_range:
    snr_lin = 10**(snr_db/10)
    N0 = Eb/snr_lin
    sigma = np.sqrt(N0/2)                                           # STEP 3
    y = s + sigma*(np.random.randn(N_sym) + 1j*np.random.randn(N_sym))
    if snr_db == sample_snr_db:
        sample_y = y
    # STEP 4: detect by quadrant
    d0 = (y.imag < 0).astype(int)
    d1 = (y.real < 0).astype(int)
    if not GRAY:
        d1 = d1 ^ d0 if False else d1       # (natural mapping decode: adjust here if you change the map)
    det = np.empty(N_bits, dtype=int); det[0::2] = d0; det[1::2] = d1
    sim_ber.append(np.sum(bits != det)/N_bits)                      # STEP 5
    th_ber.append(0.5*erfc(np.sqrt(Eb/N0)))                         # STEP 6
    print(f'SNR={snr_db:3d} dB  sim={sim_ber[-1]:.5f}  theory={th_ber[-1]:.5f}')

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
ideal = np.sqrt(Eb)*np.array([1+1j, -1+1j, -1-1j, 1-1j])
axs[0].scatter(ideal.real, ideal.imag, s=120, c='b'); axs[0].set_title('Ideal QPSK constellation')
axs[1].scatter(sample_y.real, sample_y.imag, s=1, c='r', alpha=0.5)
axs[1].scatter(ideal.real, ideal.imag, marker='x', s=100, c='b'); axs[1].set_title(f'Received (Eb/N0={sample_snr_db} dB)')
for a in axs[:2]:
    a.set_xlabel('I'); a.set_ylabel('Q'); a.grid(True); a.axhline(0, c='k', lw=.5); a.axvline(0, c='k', lw=.5)
axs[0].set_xlim(-2, 2); axs[0].set_ylim(-2, 2)
axs[2].semilogy(snr_db_range, sim_ber, 'bo-', label='Simulated')
axs[2].semilogy(snr_db_range, th_ber, 'r-', lw=2, label='Theoretical')
axs[2].set_title('QPSK BER vs SNR (AWGN)'); axs[2].set_xlabel('Eb/N0 (dB)'); axs[2].set_ylabel('BER')
axs[2].grid(True, which='both', ls='--'); axs[2].legend()
plt.tight_layout(); plt.show()

# OTHER VARIATIONS:
#  * SNR given as Es/N0: Eb = Es/2, so  Eb/N0 = Es/N0 - 3 dB.
#  * Expected: QPSK BER curve = BPSK BER curve for the same Eb/N0 (Gray mapping),
#    but QPSK sends 2 bits per symbol (half the bandwidth).
