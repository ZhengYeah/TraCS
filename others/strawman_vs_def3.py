import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
exp = np.e

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(5, 5))

bins = 6
varphi = pi / 6

epsilon = np.linspace(1.8, 10, 20, endpoint=True)
# krr
krr_p = exp ** epsilon / (exp ** epsilon + bins - 1)
krr_s = 2 * pi / bins
krr_s = np.repeat(krr_s, len(epsilon))

# circular mechanism
tracs_p = 1 / (2*pi) * exp ** (epsilon/2)
tracs_s = 2*pi * (exp ** (epsilon/2) - 1) / (exp ** epsilon - 1)
proba_s = tracs_p * tracs_s

# two y-axes plot
plt.plot(epsilon, krr_s, label='Strawman', color='red', linestyle='--')
plt.plot(epsilon, tracs_s, label='TraCS-D', color='red')
plt.yticks(np.arange(0, pi+0.2, pi/4), ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
plt.tick_params(axis='y', labelcolor='red')
plt.ylabel('Size')
plt.twinx()
plt.plot(epsilon, krr_p, label='Strawman', color='black', linestyle='--')
plt.plot(epsilon, proba_s, label='TraCS-D', color='black')
plt.ylabel('Probability')
plt.yticks(np.arange(0.4, 1.01, 0.2))
plt.xticks(np.arange(2, 11, 2))
plt.xlabel(r'Privacy parameter $\varepsilon$')

plt.savefig('strawman_vs_def3_k6.pdf', bbox_inches='tight')
plt.show()


