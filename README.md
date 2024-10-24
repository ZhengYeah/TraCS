# TraCS

## Code Structure


Class `PiecewiseMechanism` encapsulates the piecewise-based mechanisms for the circular domain and the linear domain,
and each perturbation mechanism is implemented as a method of the class. e.g.
```python
perturbed = PiecewiseMechanism(private_val, epsilon).circular_perturbation()
```
instantiates a `PiecewiseMechanism` object with the circular domain and perturbs the private value `private_val` with the given `epsilon`.


Implementation tested corrected; see directory `tests/` for the testing code.

There may be division by zero warnings or errors in running the code for discrete spaces,
generally due to the random trajectory generator generating wired trajectories.
When this happens, you can run the code again.
