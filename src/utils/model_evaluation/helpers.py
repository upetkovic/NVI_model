import pingouin as pg
import pandas as pd
import re
def estimate_icc(arr):
    """
    Estimate various ICC values for a given numpy ndarray.

    Parameters:
    - arr: numpy ndarray with shape N x M where N is the number of items and M represents the number of raters.

    Returns:
    - List of ICC values corresponding to the types computed by pingouin
    """
    
    # Dynamically create column names based on the number of raters
    num_raters = arr.shape[1]
    rater_columns = [f"Rater{i+1}" for i in range(num_raters)]
    
    # Convert the numpy ndarray to a pandas DataFrame in long format
    df = pd.DataFrame(arr, columns=rater_columns).reset_index().melt(id_vars="index", value_vars=rater_columns)
    df.columns = ["Subject", "Rater", "Rating"]
    
    # Compute the ICC
    icc_data = pg.intraclass_corr(data=df, targets="Subject", raters="Rater", ratings="Rating").set_index("Type")
    
    # Extract all ICC values
    icc_values = icc_data["ICC"].tolist()
    
    return icc_values

def parse_hidden_layers_from_filename(filename):
    """
    Extracts the hidden_layers array from a filename like 'mlp_layers_128_64_32_r_0p4903.pth'
    Returns a list of integers representing the hidden layers.
    """
    match = re.search(r"mlp_layers_([0-9_]+)_r_", filename)
    if not match:
        raise ValueError("Filename format is incorrect")
    layer_str = match.group(1)
    return [int(x) for x in layer_str.split("_")]

