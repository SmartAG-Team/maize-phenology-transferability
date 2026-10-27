"""Callable and trainable T model."""
from .._wrapper import bind

fit, predict, load, save = bind("T")

__all__ = ["fit", "predict", "load", "save"]
