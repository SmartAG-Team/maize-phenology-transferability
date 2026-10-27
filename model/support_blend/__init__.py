"""Callable and trainable support_blend model."""
from .._wrapper import bind

fit, predict, load, save = bind("support_blend")

__all__ = ["fit", "predict", "load", "save"]
