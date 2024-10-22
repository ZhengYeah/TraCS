# TraCS

## Code Structure


Class `PiecewiseMechanism` encapsulates the piecewise-based mechanisms for the circular domain and the linear domain,
and each perturbation mechanism is implemented as a method of the class. e.g.
```python
perturbed = PiecewiseMechanism(private_val, epsilon).circular_perturbation()
```
instantiates a `PiecewiseMechanism` object with the circular domain and perturbs the private value `private_val` with the given `epsilon`.


Implementation tested corrected; see directory `tests/` for the testing code.

There may be some warnings or errors in running the code for discrete spaces, 
generally due to the random trajectory generator generating wired trajectories,
i.e. $\tau_i$ and $\tau_{i+1}$ share the same value.
When this happens, you can try to run the code again.
