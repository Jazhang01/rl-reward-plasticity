import jax.numpy as jnp


def permute_observation(obs, freq=1e5):
    """
    Permute the observation deterministically using the following function:
        f(x) = sin(freq * x)
    for x in range(num_elements).

    Outputs of f(x) are sorted out-of-place, and the resulting index transformation
    will be used as the permutation for the observation.

    Assumes that obs is of shape (num_elmeents,).
    """

    assert len(obs.shape) == 1
    num_elements = obs.shape[0]

    rand_values = jnp.sin(freq * jnp.arange(1, num_elements + 1))
    indices = jnp.argsort(rand_values)

    permuted_observations = jnp.take_along_axis(obs, indices, axis=0)

    return permuted_observations
