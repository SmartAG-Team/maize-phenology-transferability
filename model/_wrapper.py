"""Bind the common interface to a single model name."""
from . import api


def bind(name):
    def fit(data, train_labels, tune_labels=None, **kwargs):
        return api.fit(name, data, train_labels, tune_labels, **kwargs)

    def predict(artifact, data, ids=None, **kwargs):
        api._artifact(artifact, name)
        return api.predict(artifact, data, ids, **kwargs)

    def load(path, **kwargs):
        return api.load(path, model=name, **kwargs)

    def save(artifact, path):
        api._artifact(artifact, name)
        return api.save(artifact, path)

    return fit, predict, load, save
