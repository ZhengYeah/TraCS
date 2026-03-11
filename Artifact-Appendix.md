# Artifact Appendix

Paper title: **TraCS: Trajectory Collection in Continuous Space under Local Differential Privacy**

Requested Badge(s):
  - [x] **Available**
  - [ ] **Functional**
  - [ ] **Reproduced**

## Description

[PETS'26] TraCS: Trajectory Collection in Continuous Space under Local Differential Privacy

Existing trajectory collection methods under LDP are largely confined to discrete location spaces, 
where the size of the location space affects both privacy guarantees and trajectory utility. 
Moreover, many real-world applications, such as flying trajectories or wearable-sensor traces,
naturally operate in continuous spaces, making these discrete-space methods inadequate.

This paper shifts the focus from discrete to continuous spaces for trajectory collection under LDP. 
We propose two methods: TraCS-D, which perturbs the direction and distance of locations, 
and TraCS-C, which perturbs the Cartesian coordinates of locations.
TraCS can also be applied to discrete spaces by rounding perturbed locations to any discrete space embedded in the continuous space. 
In this case, the privacy and utility guarantees of TraCS are independent of the number of locations in the space, 
and each perturbation requires only $\Theta(1)$ time complexity. 

This project demonstrates the advantages of TraCS by comparing its trajectory utility and time cost with those of existing methods.
### Security/Privacy Issues and Ethical Concerns

The artifact does not require any security modifications for installation or execution.
Most evaluations are theoretical or comparative in nature. The dataset included is small and publicly available, with no sensitive information involved.

## Environment

GitHub repository: https://github.com/ZhengYeah/TraCS

We did not organize the files for quick reproduction. 
However, the scripts used to generate the figures in the paper are included in the `experiments/` directory, 
and the corresponding results are stored in `experiments/results/`.
If you would like to reproduce the results in the paper, you may encounter two potential issues:
- Path configuration: Using PyCharm can simplify path configuration. If you run the scripts from the terminal, you may need to modify the import statements and relative paths in the scripts.
- Multithreading: Some scripts use multithreading to speed up execution. This reduces code flexibility and may require manually combining results from multiple runs.

## Notes on Reusability

The artifact is designed to be reusable and adaptable for trajectory collection tasks in continuous spaces under LDP. 
The main classes and methods are implemented in a hierarchical and modular manner, allowing users to easily modify the code for their specific use cases.

At the application level, users can directly use `/src/methods/wrapped_tracs.py` for trajectory collection. 
At a lower level, users can adapt the LDP mechanisms implemented in `/src/ldp_mechanisms.py`; refer to the README file for more details on the code structure and usage.