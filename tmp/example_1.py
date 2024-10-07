import math

pi = math.pi
bins = 6
epsilon = 4
varphi = pi / 6


# krr
p = math.exp(epsilon) / (math.exp(epsilon) + bins - 1)
s = (2*pi / bins)
print(f"CDF: {p:.3f}, Angular: {s/pi:.3f} pi")

# circular mechanism
p = 1 / (2*pi) * math.exp(epsilon/2)
s = 2*pi * (math.exp(epsilon/2) - 1) / (math.exp(epsilon) - 1)
l = (varphi - s / 2) / pi
r = (varphi + s / 2) / pi
proba_s = p * s
print(f"CDF: {proba_s:.3f}, Angular: {s/pi:.3f} pi")
