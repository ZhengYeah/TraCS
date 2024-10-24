import numpy as np
import pickle
import time
from src.methods.ngram import ngram_perturb
from src.methods.tp import tp_perturb
from src.methods.wrapped_tracs import tracs_d, tracs_c

pi = np.pi


def load_nyc():
    with open(f"./NYC/nyc_location_space.pkl", "rb") as f:
        location_space = pickle.load(f)
    with open(f"./NYC/nyc_trajectory.pkl", "rb") as f:
        trajectory = pickle.load(f)
    return location_space, trajectory


location_space, trajectory = load_nyc()
trajectory = trajectory[0]
length = len(trajectory)
epsilon = 2
T = 10

tp_timer_start = time.perf_counter()
for _ in range(T):
    perturbed_traj_tp = tp_perturb(trajectory, location_space, epsilon)
tp_timer_end = time.perf_counter()

ngram_timer_start = time.perf_counter()
for _ in range(3*T):
    perturbed_traj_ngram = ngram_perturb(trajectory, location_space, epsilon, theta=0.75)
ngram_timer_end = time.perf_counter()

tracs_d_timer_start = time.perf_counter()
for _ in range(T):
    perturbed_traj_tracs_d = tracs_d(trajectory, epsilon, pi / (pi + 1) * epsilon)
    for i in range(len(perturbed_traj_tracs_d)):
        nearest_location_d = location_space[np.argmin(np.linalg.norm(location_space - perturbed_traj_tracs_d[i], axis=1))]
        perturbed_traj_tracs_d[i] = nearest_location_d
tracs_d_timer_end = time.perf_counter()

tracs_c_timer_start = time.perf_counter()
for _ in range(T):
    perturbed_traj_tracs_c = tracs_c(trajectory, epsilon, epsilon / 2)
    for i in range(len(perturbed_traj_tracs_c)):
        nearest_location_c = location_space[np.argmin(np.linalg.norm(location_space - perturbed_traj_tracs_c[i], axis=1))]
        perturbed_traj_tracs_c[i] = nearest_location_c
tracs_c_timer_end = time.perf_counter()

print(f"TP Time: {(tp_timer_end - tp_timer_start)/length/T}")
print(f"NGram Time: {(ngram_timer_end - ngram_timer_start)/length/T/3}")
print(f"TRACS-D Time: {(tracs_d_timer_end - tracs_d_timer_start)/length/T}")
print(f"TRACS-C Time: {(tracs_c_timer_end - tracs_c_timer_start)/length/T}")
