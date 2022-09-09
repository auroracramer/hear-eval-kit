from joblib import delayed


def delayed_kvpair(key, val_fn):
    """Result yields a key to go with the value for convenient dictionary construction"""
    def wrapped(*args, **kwargs):
        return (key, val_fn(*args, **kwargs))

    return delayed(wrapped)