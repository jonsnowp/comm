"""
Q3: BPSK MODULATION ONLY (no channel, no detection)

PROCEDURE
 1 Take / generate a bit sequence               bits = [1,0,1,1,0,...]
 2 NRZ polar mapping                            0 -> +1, 1 -> -1  (m(t))
 3 Repeat each bit for Tb seconds               np.repeat(m, samples_per_bit)
 4 Generate carrier                             c(t) = sqrt(2Eb/Tb) cos(2 pi fc t)
 5 Multiply message by carrier                  s(t) = m(t) * c(t)
        (same as: bit 0 -> cos(2pi fc t), bit 1 -> cos(2pi fc t + pi))
 6 Plot bits, carrier, BPSK wave, constellation
"""
import numpy as np
import matplotlib.pyplot as plt

# ---- CHANGE HERE (variations) ----
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])   # or np.random.randint(0, 2, 8)
Tb = 1.0                                    # bit duration (s)
fc = 4                                      # carrier frequency (Hz) -> fc*Tb = whole cycles per bit
fs = 1000                                   # simulation sampling rate
Eb = 1

spb = int(fs*Tb)                            # samples per bit
t = np.arange(0, len(bits)*Tb, 1/fs)

m = np.repeat(np.where(bits == 0, 1, -1), spb)           # STEPS 2, 3
carrier = np.sqrt(2*Eb/Tb)*np.cos(2*np.pi*fc*t)          # STEP 4
s = m*carrier                                            # STEP 5

fig, axs = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
axs[0].step(t, np.repeat(bits, spb), where='post'); axs[0].set_title(f'Binary data {bits}'); axs[0].set_ylabel('Bit')
axs[1].plot(t, carrier); axs[1].set_title('Carrier'); axs[1].set_ylabel('Amplitude')
axs[2].plot(t, s, 'r'); axs[2].set_title('BPSK modulated signal (180 deg phase change at bit change)'); axs[2].set_ylabel('Amplitude')
for k in range(1, len(bits)):
    axs[2].axvline(k*Tb, color='k', ls='--', lw=0.5)
axs[3].scatter([np.sqrt(Eb), -np.sqrt(Eb)], [0, 0], s=120, c='b')
axs[3].set_title('BPSK constellation'); axs[3].set_xlabel('I'); axs[3].set_ylabel('Q')
axs[3].set_xlim(-2, 2); axs[3].set_ylim(-1, 1); axs[3].grid(True)
plt.show()

# OTHER VARIATIONS: change fc (e.g. 2 or 8), Tb, bit pattern, or amplitude.
#  Expected: phase of the carrier flips by 180 deg whenever the bit changes.
