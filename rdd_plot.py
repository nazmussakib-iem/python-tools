
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def rdd_plot(data, running_var, outcome_var, cutoff=0, bin_width=None,
             poly_order=1, treatment_var=None, rdd_type='sharp',
             figsize=(8, 5), title='RDD Plot', xlabel=None, ylabel=None,
             scatter_kwargs=None, line_kwargs=None, return_model=True,
             show_summary=True):
    """
    Regression Discontinuity Design plot with regression results (sharp/fuzzy).

    Parameters:
    - data: DataFrame containing the data
    - running_var: Name of the running variable (forcing variable)
    - outcome_var: Name of the outcome variable
    - cutoff: Threshold value for treatment assignment
    - bin_width: Width of bins for plotting means (auto-calculated if None)
    - poly_order: Polynomial order of running variable
    - treatment_var: Actual treatment variable (required for fuzzy RDD)
    - rdd_type: 'sharp' or 'fuzzy'
    - figsize: Plot size
    - title: Plot title
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - scatter_kwargs: Dict of kwargs for scatterplot
    - line_kwargs: Dict of kwargs for fitted line
    - return_model: If True, returns regression model
    - show_summary: If True, prints regression summaries
    """

    scatter_kwargs = scatter_kwargs or {'color': 'gray', 'alpha': 0.5}
    line_kwargs = line_kwargs or {'linewidth': 2}

    data = data.dropna(subset=[running_var, outcome_var])
    data['running_centered'] = data[running_var] - cutoff

    if bin_width is None:
        q75, q25 = np.percentile(data[running_var], [75, 25])
        iqr = q75 - q25
        n = len(data)
        bin_width = 2 * iqr / (n ** (1/3)) if iqr > 0 else (data[running_var].max() - data[running_var].min()) / 20

    data['bin'] = (data['running_centered'] / bin_width).round() * bin_width
    binned = data.groupby('bin')[[outcome_var]].mean().reset_index()

    for i in range(1, poly_order + 1):
        data[f'running^{i}'] = data['running_centered'] ** i

    formula_terms = [f'running^{i}' for i in range(1, poly_order + 1)]
    data['treat'] = (data['running_centered'] >= 0).astype(int)

    if rdd_type == 'sharp':
        X = sm.add_constant(data[['treat'] + formula_terms])
        model = sm.OLS(data[outcome_var], X).fit()
        if show_summary:
            print("\n=== SHARP RDD: Regression Summary ===")
            print(model.summary())

    elif rdd_type == 'fuzzy':
        if treatment_var is None:
            raise ValueError("For fuzzy RDD, you must provide treatment_var.")

        data = data.dropna(subset=[treatment_var])

        # First stage
        X1 = sm.add_constant(data[['treat'] + formula_terms])
        y1 = data[treatment_var]
        stage1 = sm.OLS(y1, X1).fit()

        # Second stage
        data['predicted_treatment'] = stage1.predict(X1)
        X2 = sm.add_constant(pd.concat([data['predicted_treatment'], data[formula_terms]], axis=1))
        y2 = data[outcome_var]
        stage2 = sm.OLS(y2, X2).fit()

        model = stage2

        if show_summary:
            print("\n=== FUZZY RDD: First Stage (Treatment ~ Instrument) ===")
            print(stage1.summary())
            print("\n=== FUZZY RDD: Second Stage (Outcome ~ Predicted Treatment) ===")
            print(stage2.summary())
    else:
        raise ValueError("rdd_type must be 'sharp' or 'fuzzy'")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(binned['bin'], binned[outcome_var], label='Binned Averages', **scatter_kwargs)
    ax.axvline(0, color='red', linestyle='--', label='Cutoff')

    x_vals = np.linspace(data['running_centered'].min(), data['running_centered'].max(), 200)
    x_df = pd.DataFrame({'running_centered': x_vals})
    for i in range(1, poly_order + 1):
        x_df[f'running^{i}'] = x_vals ** i
    x_df['treat'] = (x_vals >= 0).astype(int)
    x_df = sm.add_constant(x_df)

    if rdd_type == 'sharp':
        x_pred = x_df[['const', 'treat'] + formula_terms]
        y_pred = model.predict(x_pred)
    else:
        x_pred1 = x_df[['const', 'treat'] + formula_terms]
        treat_hat = stage1.predict(x_pred1)
        x_pred2 = pd.concat([treat_hat.rename('predicted_treatment'), x_df[formula_terms]], axis=1)
        x_pred2 = sm.add_constant(x_pred2)
        y_pred = stage2.predict(x_pred2)

    ax.plot(x_vals, y_pred, color='blue', label='Fitted Line', **line_kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel or running_var)
    ax.set_ylabel(ylabel or outcome_var)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    return (ax, model) if return_model else ax
