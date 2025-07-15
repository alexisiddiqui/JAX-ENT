def determine_optimal_backend(backend: str = "auto", verbose: bool = True) -> str:
    """
    Determine the optimal backend for MDAnalysis distance calculations.

    Parameters
    ----------
    backend : str, default "auto"
        Requested backend: 'serial', 'OpenMP', or 'auto' for automatic detection
    verbose : bool, default True
        Whether to print backend selection information

    Returns
    -------
    str
        The backend to use ('serial' or 'OpenMP')
    """
    if backend == "auto":
        try:
            # Check if OpenMP is available
            from MDAnalysis.lib import distances as lib_distances

            if hasattr(lib_distances, "USED_OPENMP") and lib_distances.USED_OPENMP:
                backend = "OpenMP"
                if verbose:
                    print("Using OpenMP backend for distance calculations")
            else:
                backend = "serial"
                if verbose:
                    print("OpenMP not available, using serial backend")
        except (ImportError, AttributeError):
            backend = "serial"
            if verbose:
                print("OpenMP detection failed, using serial backend")

    return backend
