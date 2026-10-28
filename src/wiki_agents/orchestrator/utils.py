import os


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer from the environment.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is not set.

    Returns:
        A positive integer.

    """
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    """Read a nonnegative integer from the environment.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is not set.

    Returns:
        A nonnegative integer.

    """
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value
