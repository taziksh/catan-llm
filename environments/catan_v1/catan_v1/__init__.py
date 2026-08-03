from catan_v1.legacy import load_hosted_environment
from catan_v1.taskset import CatanEnv, CatanTaskset

__all__ = ["CatanEnv", "CatanTaskset", "load_environment"]


def load_environment(**kwargs):
    """Load the Prime Hosted Training-compatible environment."""
    return load_hosted_environment(**kwargs)
