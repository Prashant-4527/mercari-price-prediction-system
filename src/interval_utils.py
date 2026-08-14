import numpy as np

def get_prediction_interval(pred_log, bin_edges, bin_bounds):
    """Given a point prediction (log-space) and pre-computed bin boundaries,
    return a (lower_price, upper_price) dollar interval."""
    bin_idx = np.digitize(pred_log, bin_edges[1:-1])
    bin_idx = min(bin_idx, len(bin_bounds) - 1) # clip if beyond the last known bin

    lower_pct, upper_pct = bin_bounds[bin_idx]
    lower_log = pred_log + lower_pct
    upper_pct = pred_log + upper_pct


    return float(np.expm1(lower_log)), float(np.expm1(upper_pct))


