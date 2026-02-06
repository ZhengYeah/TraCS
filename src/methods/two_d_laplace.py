""""
2D Local Differential Privacy for points in [0,1]^2 via Planar Laplace Mechanism
"""
import math
import random
from typing import  List, Optional, Sequence, Tuple

Point2D = Tuple[float, float]

def sample_planar_laplace(eps: float, rng: Optional[random.Random] = None) -> Point2D:
    """
    Sample Z ~ planar (isotropic) Laplace in R^2 with density:
        f(z) = (eps^2 / (2*pi)) * exp(-eps * ||z||_2)
    Sampling method:
        R ~ Gamma(k=2, scale=1/eps), Theta ~ Uniform[0, 2*pi)
        Z = (R cos Theta, R sin Theta)
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")

    rng = rng or random

    # Gamma(k=2, theta=1/eps) can be sampled as sum of two iid Exp(rate=eps).
    # Exp(rate=eps): -log(U)/eps
    u1 = rng.random()
    u2 = rng.random()
    r = (-math.log(u1) - math.log(u2)) / eps

    theta = 2.0 * math.pi * rng.random()
    return r * math.cos(theta), r * math.sin(theta)


def clip_to(y: Point2D) -> Point2D:
    """Project to [0, low] \times [0, high] via coordinate-wise clipping (post-processing)."""
    return min(1.0, max(0.0, y[0])), min(1.0, max(0.0, y[1]))



def ldp_2d_laplace(x: Point2D, eps: float, clip: bool = False, rng: Optional[random.Random] = None) -> Point2D:
    """
    Return M(x) = x + Z where Z ~ planar Laplace(eps). Optionally clip to [0,1]^2.
    epsilon: for [0,1]^2 domain, actual epsilon is eps/sqrt(2)
    """
    eps = eps / math.sqrt(2) # adjust for L2 sensitivity of 2D points in [0,1]^2
    z = sample_planar_laplace(eps, rng=rng)
    y = (x[0] + z[0], x[1] + z[1])
    return clip_to(y) if clip else y


def ldp_2d_laplace_batch(xs: Sequence[Point2D], eps: float, clip: bool = True, seed: Optional[int] = None) -> List[Point2D]:
    """
    Batch version with optional deterministic seeding.
    Usage: ldp_2d_laplace_batch(trajectory, epsilon, clip=True)
    """
    rng = random.Random(seed) if seed is not None else None
    return [ldp_2d_laplace(x, eps, clip=clip, rng=rng) for x in xs]


if __name__ == "__main__":
    x = (0.3, 0.8)
    eps = 1.0

    for _ in range(5):
        print("no clip:", ldp_2d_laplace(x, eps, clip=False))
    for _ in range(5):
        print("clipped:", ldp_2d_laplace(x, eps, clip=True))
