
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

N_bits = 200000
Eb = 1
snr_db_range = np.arange(-2, 11)
sample_snr_db = 5

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

bits = np.random.randint(0, 2, N_bits)
bit_pairs = bits.reshape(-1, 2)

N_symbols = len(bit_pairs)
s = np.zeros(N_symbols, dtype=complex)

for i in range(N_symbols):
    b0, b1 = bit_pairs[i]

    if b0 == 0 and b1 == 0:
        s[i] = np.sqrt(Eb) * (1 + 1j)
    elif b0 == 0 and b1 == 1:
        s[i] = np.sqrt(Eb) * (-1 + 1j)
    elif b0 == 1 and b1 == 1:
        s[i] = np.sqrt(Eb) * (-1 - 1j)
    elif b0 == 1 and b1 == 0:
        s[i] = np.sqrt(Eb) * (1 - 1j)

axs[0].scatter(s.real, s.imag, color='blue', label='Ideal QPSK')
axs[0].set_title('Ideal QPSK Constellation')
axs[0].set_xlabel('In-Phase (I)')
axs[0].set_ylabel('Quadrature (Q)')
axs[0].grid(True)
axs[0].axhline(0, color='black', lw=0.5)
axs[0].axvline(0, color='black', lw=0.5)
axs[0].set_xlim(-2, 2)
axs[0].set_ylim(-2, 2)
axs[0].legend()

simulated_ber = []
theoretical_ber = []
sample_y = None

for snr_db in snr_db_range:
    snr_lin = 10 ** (snr_db / 10)
    N0 = Eb / snr_lin
    sigma = np.sqrt(N0 / 2)

    noise = sigma * (
        np.random.randn(N_symbols) +
        1j * np.random.randn(N_symbols)
    )

    y = s + noise

    if snr_db == sample_snr_db:
        sample_y = y

    detected_bits = np.zeros(N_bits, dtype=int)

    for i in range(N_symbols):
        re = np.real(y[i])
        im = np.imag(y[i])

        if re >= 0 and im >= 0:
            detected_bits[2*i:2*i+2] = [0, 0]
        elif re < 0 and im >= 0:
            detected_bits[2*i:2*i+2] = [0, 1]
        elif re < 0 and im < 0:
            detected_bits[2*i:2*i+2] = [1, 1]
        elif re >= 0 and im < 0:
            detected_bits[2*i:2*i+2] = [1, 0]

    errors = np.sum(bits != detected_bits)
    simulated_ber.append(errors / N_bits)

    P_b = 0.5 * erfc(np.sqrt(Eb / N0))
    theoretical_ber.append(P_b)

axs[1].scatter(
    sample_y.real, sample_y.imag,
    s=1, color='red', alpha=0.5,
    label=f'Received (SNR={sample_snr_db}dB)'
)

ideal_points = np.array([
    1+1j, -1+1j, -1-1j, 1-1j
]) * np.sqrt(Eb)

axs[1].scatter(
    ideal_points.real, ideal_points.imag,
    color='blue', marker='x', s=100,
    label='Ideal Points'
)

axs[1].set_title(
    f'QPSK Received Constellation (Eb/N0 = {sample_snr_db} dB)'
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

axs[2].set_title('QPSK Error Performance over AWGN Channel')
axs[2].set_xlabel('(Eb/N0) (dB)')
axs[2].set_ylabel('Bit Error Rate (BER)')
axs[2].grid(True, which='both', ls='--')
axs[2].legend()

plt.tight_layout()
plt.show()
