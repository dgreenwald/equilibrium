def initialize_if_none(x, default):
    """
    Initialize x to default if x is None.

    Parameters
    ----------
    x : object
        Object to be initialized.
    default : object
        Default value for x.
    """
    if x is None:
        return default
    else:
        return x
