
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def rdd_plot(data, outcome_variable, running_variable, cutoff, fuzzy=False, order=1, bandwidth=None, plot_type='scatter'):
    if bandwidth is None:
        bandwidth = np.std(data[running_variable]) * 0.5  # automatic bandwidth

    data_trimmed = data[(data[running_variable] >= cutoff - bandwidth) & 
                        (data[running_variable] <= cutoff + bandwidth)].copy()

    data_trimmed['treated'] = (data_trimmed[running_variable] >= cutoff).astype(int)

    for i in range(2, order+1):
        data_trimmed[f'{running_variable}^{i}'] = data_trimmed[running_variable] ** i

    predictors = [running_variable] + [f'{running_variable}^{i}' for i in range(2, order+1)] + ['treated']
    X = sm.add_constant(data_trimmed[predictors])
    y = data_trimmed[outcome_variable]

    model = sm.OLS(y, X).fit()

    fig, ax = plt.subplots(figsize=(8, 5))
    left = data_trimmed[data_trimmed[running_variable] < cutoff]
    right = data_trimmed[data_trimmed[running_variable] >= cutoff]

    ax.scatter(left[running_variable], left[outcome_variable], color='royalblue', alpha=0.6, label='Left of cutoff')
    ax.scatter(right[running_variable], right[outcome_variable], color='darkred', alpha=0.6, label='Right of cutoff')

    x_vals = np.linspace(data_trimmed[running_variable].min(), data_trimmed[running_variable].max(), 300)
    df_pred = pd.DataFrame({running_variable: x_vals})
    for i in range(2, order+1):
        df_pred[f'{running_variable}^{i}'] = x_vals ** i
    df_pred['treated'] = (x_vals >= cutoff).astype(int)
    X_pred = sm.add_constant(df_pred[predictors])
    y_pred = model.predict(X_pred)

    ax.plot(x_vals[x_vals < cutoff], y_pred[x_vals < cutoff], color='navy', linewidth=2.5, label='Fit (Left)')
    ax.plot(x_vals[x_vals >= cutoff], y_pred[x_vals >= cutoff], color='darkorange', linewidth=2.5, label='Fit (Right)')

    ax.axvline(cutoff, color='black', linestyle='--', linewidth=1.5, label='Cutoff')
    ax.set_xlabel(running_variable)
    ax.set_ylabel(outcome_variable)
    ax.set_title("Regression Discontinuity Plot")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return model.summary()
