"""Callable and trainable thermal_transformer model."""
from .._wrapper import bind

fit, predict, load, save = bind("thermal_transformer")

__all__ = ["fit", "predict", "load", "save"]
