"""Callable and trainable residual_RF model."""
from .._wrapper import bind

fit, predict, load, save = bind("residual_RF")

__all__ = ["fit", "predict", "load", "save"]
