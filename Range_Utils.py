import numpy as np

def change_range(data, oldMin_Range,  oldMax_Range, newMin_Range, newMax_Range):
    """
    Maps a number, list, or Numpy array from OneRange[oldMin_Range, oldMax_Range] to Another[newMin_Range, newMax_Range].

    Parameters:
        data: scaler, list, or np.ndarray - the input value(s)
        oldMin_Range: float – original minimum of the data
        oldMax_Range: float – original maximum of the data
        newMin_Range: float – new desired minimum
        newMax_Range: float – new desired maximum

    Return:
        Scaled data in the new range.
    """
    try:
        # Convert list or scalar to numpy array
        data = np.array(data, dtype=np.float32)
        oldMin_Range = float(oldMin_Range)
        oldMax_Range = float(oldMax_Range)
        newMin_Range = float(newMin_Range)
        newMax_Range = float(newMax_Range)
        scaled_data = ((data - oldMin_Range) / (oldMax_Range - oldMin_Range)) * (newMax_Range - newMin_Range) + newMin_Range

        return scaled_data

    except Exception as e:
        print("Error in range_change:", e)
        raise ValueError("Input data could not be scaled. Ensure it's a number, list, or NumPy array.")
