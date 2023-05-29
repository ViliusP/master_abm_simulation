import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def calculate_weight_distribution_kde(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Gaussian KDE for weight distribution and adds a 'Probability' column to a copy of the input DataFrame.
    This function also filters out zero weights and removes outliers.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing weight information.

    Returns:
    pd.DataFrame: A copy of the input DataFrame with an added 'Probability' column.
    """
    # Zero weights filer
    non_zero_weights = data[data['Weight'] > 0]

    q1 = non_zero_weights['Weight'].quantile(0.25)
    q3 = non_zero_weights['Weight'].quantile(0.75)
    iqr = q3 - q1

    # Bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter out outliers
    data_copy = non_zero_weights[(non_zero_weights['Weight'] >= lower_bound) & (non_zero_weights['Weight'] <= upper_bound)].copy()

    weights = data_copy['Weight']

    # Calculate the Gaussian KDE
    kernel = gaussian_kde(weights)

    # Apply the kernel to each weight to get its probability
    data_copy['Probability'] = data_copy['Weight'].apply(kernel)
    print(data_copy.head(10))
    return data_copy


def sample_weight(data: pd.DataFrame) -> float:
    """
    Randomly samples a weight from the given DataFrame.

    Parameters:
    data (pd.DataFrame): A DataFrame with 'Weight' and 'Probability' columns.

    Returns:
    float: A randomly sampled weight from the DataFrame.
    """
    return data.sample(n=1, weights='Probability').iloc[0]['Weight']
 