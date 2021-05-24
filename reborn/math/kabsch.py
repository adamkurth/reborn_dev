import scipy.spatial.transform


def kabsch(A, A0):
    r"""
    Finds the rotation matrix that will bring a set of vectors A into alignment with 
    another set of vectors A0. Uses the Kabsch algorithm implemented in 
    scipy.spatial.transform.Rotation.align_vectors

    Arguments:
        A (|ndarray|): N, 3x1 vectors stacked into the shape (N,3) 
        A0 (|ndarray|): N, 3x1 vectors stacked into the shape (N,3) 

    Returns:
        |ndarray|: 3x3 rotation matrix.
    """
    return scipy.spatial.transform.Rotation.align_vectors(A0, A)[0].as_matrix()
