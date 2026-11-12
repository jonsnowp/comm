"""
Q7: AM MODULATOR - observe the waveform and calculate the modulation index

PROCEDURE
 1 Message signal            m(t) = Am cos(2 pi fm t)
 2 Carrier signal            c(t) = Ac cos(2 pi fc t)        (fc >> fm)
 3 AM (DSB-FC) wave          s(t) = Ac [1 + mu cos(2 pi fm t)] cos(2 pi fc t)
                             where mu = Am/Ac   (theoretical modulation index)
 4 Envelope                  |hilbert(s)|  (or Ac*(1 + mu cos(...)))
 5 Measure modulation index  mu = (Emax - Emin) / (Emax + Emin)
 6 Plot message, carrier, AM wave with envelope
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ---- CHANGE HERE (variations) ----
Ac, Am = 1.0, 0.5                # mu = Am/Ac = 0.5.  Try Am = 1.0 (mu=1) and Am = 1.5 (mu>1, over-modulation)
fm, fc = 10, 200                 # Hz.  fc must be >> fm
fs = 20000
t = np.arange(0, 0.3, 1/fs)

m = Am*np.cos(2*np.pi*fm*t)                      # STEP 1
c = Ac*np.cos(2*np.pi*fc*t)                      # STEP 2
s = (Ac + m)*np.cos(2*np.pi*fc*t)                # STEP 3  = Ac(1 + mu cos) cos
# DSB-SC variation:  s = m*np.cos(2*np.pi*fc*t)  (no carrier term, no mu)

env = np.abs(hilbert(s))                         # STEP 4
E_max, E_min = np.max(env), np.min(env)
mu_theory = Am/Ac
mu_measured = (E_max - E_min)/(E_max + E_min)    # STEP 5
print(f'Theoretical mu = Am/Ac = {mu_theory:.3f}')
print(f'Measured    mu = (Emax-Emin)/(Emax+Emin) = {mu_measured:.3f}   (Emax={E_max:.3f}, Emin={E_min:.3f})')
if mu_theory < 1:   print('Under-modulation (mu<1): envelope follows message')
elif mu_theory == 1: print('100% modulation (mu=1)')
else:               print('Over-modulation (mu>1): envelope distortion, phase reversals')

fig, axs = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
axs[0].plot(t, m); axs[0].set_title(f'Message m(t), fm={fm} Hz, Am={Am}')
axs[1].plot(t, c); axs[1].set_title(f'Carrier c(t), fc={fc} Hz, Ac={Ac}')
axs[2].plot(t, s, label='AM wave'); axs[2].plot(t, env, 'r', label='Envelope'); axs[2].plot(t, -env, 'r')
axs[2].set_title(f'AM wave, mu = {mu_measured:.2f}'); axs[2].legend()
for a in axs: a.set_xlabel('Time (s)'); a.set_ylabel('Amplitude'); a.grid(True)
plt.show()

# OTHER VARIATIONS:
#  * Loop over Am = [0.5, 1.0, 1.5] to show under / critical / over-modulation.
#  * Percentage modulation = mu * 100 %.
#  * Total power Pt = Pc (1 + mu^2/2), sideband power = Pc*mu^2/2, Pc = Ac^2/2.
