"""Callable and trainable nonlinear_T model."""
from .._wrapper import bind

fit, predict, load, save = bind("nonlinear_T")

__all__ = ["fit", "predict", "load", "save"]
