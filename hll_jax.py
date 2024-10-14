import jax.numpy as jnp
from jax import random
import timeit

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(5.0)
print(selu(x))

key = random.PRNGKey(1701)
x = random.normal(key, (1_000_000,))

# Define a function to time
def time_selu():
    selu(x).block_until_ready()

# Use timeit to measure the execution time
execution_time = timeit.timeit(time_selu, number=10)
print(f"Average execution time over 10 runs: {execution_time / 10:.6f} seconds")