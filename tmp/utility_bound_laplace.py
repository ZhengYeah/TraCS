import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace
from matplotlib.ticker import FormatStrFormatter


# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20


epsilon = [1, 5, 10]

# Parameters for the Laplace distribution
mu = 0  # Location parameter
b = np.ones(len(epsilon)) / np.array(epsilon)  # Scale parameter

# CDF list
theta = np.linspace(0, 0.3, 100)
cdf_list = np.zeros([len(epsilon), len(theta)])
for i in range(len(epsilon)):
    cdf = laplace.cdf(theta, mu, b[i])
    p = 2 * cdf - 1
    cdf_list[i] = p

# Plot cdf list
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.plot(theta, cdf_list[0], 'black', label=r'$\varepsilon=1$', linewidth=2)
plt.plot(theta, cdf_list[1], 'blue', label=r'$\varepsilon=5$', linewidth=2)
plt.plot(theta, cdf_list[2], 'red', label=r'$\varepsilon=10$', linewidth=2)

# Add labels and title
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\varepsilon, \theta)$')
plt.ylim(0, 1)
plt.xlim(0, 0.35)
plt.legend()

# Remove ending zeros in the ticks
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))


# Show the plot
plt.savefig('./utility_bound_laplace.pdf')
plt.show()
