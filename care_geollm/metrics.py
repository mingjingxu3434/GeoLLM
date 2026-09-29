import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

TARGET_NAMES = [
    'accessibility_gain', 'mobility_burden_reduction',
    'operational_carbon_reduction', 'carbon_payback_improvement'
]

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    per_target = {}
    for j, name in enumerate(TARGET_NAMES[:y_true.shape[1]]):
        yt, yp = y_true[:, j], y_pred[:, j]
        rho = spearmanr(yt, yp).statistic if len(yt) > 1 else np.nan
        per_target[name] = {
            'MAE': float(mean_absolute_error(yt, yp)),
            'RMSE': float(mean_squared_error(yt, yp) ** 0.5),
            'R2': float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
            'Spearman': float(rho) if np.isfinite(rho) else np.nan,
        }
    flat_t, flat_p = y_true.reshape(-1), y_pred.reshape(-1)
    rho = spearmanr(flat_t, flat_p).statistic if flat_t.size > 1 else np.nan
    overall = {
        'MAE': float(mean_absolute_error(flat_t, flat_p)),
        'RMSE': float(mean_squared_error(flat_t, flat_p) ** 0.5),
        'R2': float(r2_score(flat_t, flat_p)) if flat_t.size > 1 else np.nan,
        'Spearman': float(rho) if np.isfinite(rho) else np.nan,
    }
    return {'overall': overall, 'per_target': per_target}
