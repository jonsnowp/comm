
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

N_bits = 100000
Eb = 1
snr_db_range = np.arange(-2, 11)
sample_snr_db = 5

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

bits = np.random.randint(0, 2, N_bits)
s = np.where(bits == 0, np.sqrt(Eb), -np.sqrt(Eb))

axs[0].scatter(s.real, s.imag, color='blue', label='Ideal BPSK')
axs[0].set_title('Ideal BPSK Constellation')
axs[0].set_xlabel('In-Phase (I)')
axs[0].set_ylabel('Quadrature (Q)')
axs[0].grid(True)
axs[0].axhline(0, color='black', lw=0.5)
axs[0].axvline(0, color='black', lw=0.5)
axs[0].legend()

simulated_ber = []
theoretical_ber = []
sample_y = None

for snr_db in snr_db_range:
    snr_lin = 10 ** (snr_db / 10)
    N0 = Eb / snr_lin
    sigma = np.sqrt(N0 / 2)

    noise = sigma * (
        np.random.randn(N_bits) +
        1j * np.random.randn(N_bits)
    )

    y = s + noise

    if snr_db == sample_snr_db:
        sample_y = y

    detected_bits = np.where(np.real(y) >= 0, 0, 1)

    errors = np.sum(bits != detected_bits)
    simulated_ber.append(errors / N_bits)

    P_b = 0.5 * erfc(np.sqrt(Eb / N0))
    theoretical_ber.append(P_b)

axs[1].scatter(
    sample_y.real, sample_y.imag,
    s=1, color='red', alpha=0.5,
    label=f'Received (SNR={sample_snr_db}dB)'
)

axs[1].scatter(
    [np.sqrt(Eb), -np.sqrt(Eb)],
    [0, 0],
    color='blue', marker='x', s=100,
    label='Ideal Points'
)

axs[1].set_title(
    f'BPSK Received Constellation (Eb/N0 = {sample_snr_db} dB)'
)
axs[1].set_xlabel('In-Phase (I)')
axs[1].set_ylabel('Quadrature (Q)')
axs[1].grid(True)
axs[1].axhline(0, color='black', lw=0.5)
axs[1].axvline(0, color='black', lw=0.5)
axs[1].legend()

axs[2].semilogy(
    snr_db_range, simulated_ber, 'bo-',
    label='Simulated BER'
)

axs[2].semilogy(
    snr_db_range, theoretical_ber, 'r-',
    linewidth=2, label='Theoretical BER'
)

axs[2].set_title('BPSK Error Performance over AWGN Channel')
axs[2].set_xlabel('(Eb/N0) (dB)')
axs[2].set_ylabel('Bit Error Rate (BER)')
axs[2].grid(True, which='both', ls='--')
axs[2].legend()

plt.tight_layout()
plt.show()
