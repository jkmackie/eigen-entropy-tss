import numpy as np
import pandas as pd

def confusion_matrix_values(y_true: bool , y_hat: bool):
    '''Get binary confusion matrix results.  Compute accuracy, precision,
    recall, and f1 score.
    
    Returns
    TP: int count of True Positives
    FP: int count of Fasle Positives
    TN: int count of True Negatives
    FN: int count of False Positives
    '''
    import numpy as np
    TP = np.sum((y_true == 1) & (y_hat == 1))
    FP = np.sum((y_true == 0) & (y_hat == 1))
    TN = np.sum((y_true == 0) & (y_hat == 0))
    FN = np.sum((y_true == 1) & (y_hat == 0))
    print("TP:", TP, "|| FP:",  FP, "|| TN:", TN, "|| FN:", FN, "|| Total:", TP+FP+TN+FN)
    print('   recall:', round(TP/(TP+FN),4))
    print('precision:', round(TP/(TP+FP),4))
    print(' accuracy:', round((TP+TN)/(TP+TN+FP+FN),4))    
    print('       f1:', round(2*TP/(2*TP+FP+FN),4))
    print('      CSI:', round(TP/(TP+FP+FN),4)) # ignores true negatives due to rare positives
    
    return TP, FP, TN, FN

def coarse_time_series(X: pd.Series | pd.DataFrame, scale: int | float):
    """Extract coarse-grained time series from X.  Pandas format
    enforced to work with eigen_entropy.

    Parameters
    ----------
    X : 1D pd.Series, 2D pd.Dataframe
        Time matrix of shape (rows x cols). rows are observations
        at time1, time2, time3, etc.  columns are features.
    scale : int, float
        Consecutive points to aggregate

    Returns
    -------
    coarse_ts : pd.Series if 1D, pd.DataFrame if 2D
                Coarse-grained time series with a given scale factor
    """
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        print("X must be a pd.Series or pd.Dataframe with text column names.")
    time_matrix = np.asarray(X)

    # Reshape matrix.  Avg the scale dim.
    ts_len = len(time_matrix)   # 1st dim len
    ts_len= ts_len // scale         # trunc excess ts_len elements
    if time_matrix.ndim == 1:   # reshape(rows, scale)
        coarse_ts = np.mean(time_matrix[: ts_len * scale].reshape(ts_len, scale), axis=1)
        return pd.Series(data=coarse_ts, name=X.name)
    elif time_matrix.ndim == 2:                           # reshape(rows, scale, cols)
        coarse_ts_2D = np.mean(time_matrix[: ts_len * scale].reshape(ts_len, scale, time_matrix.shape[1]), axis=-2)  
        return pd.DataFrame(data=coarse_ts_2D, columns=X.columns)
    else:
        print("coarse_time_series() handles 1D and 2D matrices only.")