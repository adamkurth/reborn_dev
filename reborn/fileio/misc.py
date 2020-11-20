import pickle


def load_pickle(pickle_file):
    r""" Open a pickle file.
    Arguments:
        pickle_file (str): Path to the pickle file
    Returns: data
    """
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pickle_file):
    r""" Save something in pickle format.
    Arguments:
        data (anything, usually a dict): The data you want to pickle
        pickle_file (str): Path to the pickle file
    """
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
