"""
Q8: EYE DIAGRAM FOR DIFFERENT ROLL-OFF FACTORS

PROCEDURE
 1 Generate bits -> polar symbols (0 -> +1, 1 -> -1)
 2 Upsample by L (insert L-1 zeros)
 3 Generate SRRC pulse for each roll-off alpha
 4 Pulse shape: convolve with p                 s = conv(upsampled, p)
 5 (optional) add AWGN                          noise = sqrt(mean(s^2)/snr_lin) * randn
 6 Matched filter: convolve with p reversed     y = conv(r, p[::-1])
 7 Cut y into traces of EYE_SPAN symbols, starting at the first symbol instant (len(p)-1)
 8 Overlay all traces -> eye diagram, one subplot per alpha
"""
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
            p[i] = (alpha/np.sqrt(2))*((1 + 2/np.pi)*np.sin(np.pi/(4*alpha)) + (1 - 2/np.pi)*np.cos(np.pi/(4*alpha)))
        else:
            p[i] = (np.sin(np.pi*tc*(1-alpha)) + 4*alpha*tc*np.cos(np.pi*tc*(1+alpha))) / (np.pi*tc*(1-(4*alpha*tc)**2))
    return p

# ---- CHANGE HERE (variations) ----
alphas = [0.1, 0.25, 0.5, 0.9]       # roll-off factors to compare
SNR_DB = None                        # None = no noise ; e.g. 8 or 4 to add noise
EYE_SPAN = 3                         # symbols per trace (2 is the textbook eye; lab code used 3)
N_bits, L, N_sym, nTraces = 2000, 4, 8, 100

bits = np.random.randint(0, 2, N_bits)
symbols = np.where(bits == 0, 1, -1)
upsampled = np.zeros(len(symbols)*L); upsampled[::L] = symbols

fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for ax, alpha in zip(axs.flatten(), alphas):
    p = generate_srrc(alpha, N_sym, L)                                  # STEP 3
    s = np.convolve(upsampled, p, mode='full')                          # STEP 4
    r = s if SNR_DB is None else s + np.sqrt(np.mean(s**2)/10**(SNR_DB/10))*np.random.randn(len(s))  # STEP 5
    y = np.convolve(r, p[::-1], mode='full')                            # STEP 6
    start = len(p) - 1                                                  # STEP 7
    nS = EYE_SPAN*L
    for j in range(nTraces):
        idx = start + j*L
        if idx + nS < len(y):
            ax.plot(np.arange(nS)/L, y[idx:idx+nS], 'b-', alpha=0.3)    # STEP 8
    ax.set_title(f'Eye diagram, roll-off alpha = {alpha}')
    ax.set_xlabel('Time (symbol periods)'); ax.set_ylabel('Amplitude'); ax.grid(True)
plt.show()

# EXPECTED: small alpha -> eye opens only at the exact sampling instant (sensitive to timing error);
#           large alpha -> wider eye, less ISI but more bandwidth (BW = (1+alpha)Rs/2).
