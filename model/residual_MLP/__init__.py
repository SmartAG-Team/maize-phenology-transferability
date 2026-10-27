"""Callable and trainable residual_MLP model."""
from .._wrapper import bind

fit, predict, load, save = bind("residual_MLP")

__all__ = ["fit", "predict", "load", "save"]
