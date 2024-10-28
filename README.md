# TraCS

## Reproductions

To reproduce the results in the paper, you can change to the `experiments/continuous_space` directory
or the `experiments/discrete_space` directory and run the corresponding script, e.g.
```bash
python experiment_1_1.py
```
will print the results of the first experiment in the paper.
The `csv`files containing the results will be saved in the `results/` directory, 
where also contains the drawing scripts for the figures in the paper. 


## Code Structure

The code is organized as follows:
- `src/` contains the source code for the TraCS mechanisms and other methods.
    - `src/ldp_mechanisms.py` contains the LDP mechanisms used in this paper.
    - `src/perturbation_tracs.py` contains the implementation of TraCS-D and TraCS-C.
    - `src/methods.py` wraps the TraCS mechanisms and other methods.
    - `src/utilities.py` contains some utility functions.
- `tests/` contains the testing code for the TraCS mechanisms and other methods.
- `experiments/` contains the code for the experiments in the paper.

In file `src/ldp_mechanisms.py`, we classify the LDP mechanisms into `PiecewiseMechanism` and `DiscreteMechanism` classes.
Class `PiecewiseMechanism` encapsulates the piecewise-based mechanisms for the circular domain and the linear domain,
and each perturbation mechanism is implemented as a method of the class. e.g.
```python
perturbed_direction = PiecewiseMechanism(private_val, epsilon).circular_perturbation()
```
instantiates a `PiecewiseMechanism` object with the circular domain and perturbs the private value `private_val` with the given `epsilon`.

In file `src/perturbation_tracs.py`, the
TraCS-D and TraCS-C are encapsulated as classes `DirectionDistancePerturbation` and `CoordinatePerturbation`.


Implementation tested corrected; see directory `tests/` for the testing code.

There may be division by zero warnings or errors in running the code for discrete spaces,
generally due to the random trajectory generator generating wired trajectories.
When this happens, you can run the code again.
