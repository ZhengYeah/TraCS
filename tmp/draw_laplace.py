import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace

# Parameters for the Laplace distribution
mu = 0.5  # Location parameter
b = 0.1   # Scale parameter

# Define the PDF of the Laplace distribution
x = np.linspace(-0.5, 1.5, 1000)
pdf = laplace.pdf(x, mu, b)

# Plot the PDF of the Laplace distribution
plt.plot(x, pdf, 'black')

# Add labels and title
plt.xlabel('x')
plt.ylim(0, 5)

# Show the plot
plt.savefig('./laplace.pdf')
plt.show()
