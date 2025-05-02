
import matplotlib.pyplot as plt
import pandas as pd

def coefplot2(model, exclude_intercept=True, figsize=(8, 5), title='Coefficient Plot', show_values=True):
    """
    Plots coefficient estimates and 95% confidence intervals from a statsmodels model.
    
    Parameters:
    - model: A fitted statsmodels model (e.g., OLS, Logit)
    - exclude_intercept: Whether to exclude 'const' (intercept) from plot
    - figsize: Tuple for figure size
    - title: Title of the plot
    - show_values: Whether to annotate the coefficients above points
    """
    # Extract coefficient estimates and confidence intervals
    params = model.params
    conf = model.conf_int()
    
    df_plot = pd.DataFrame({
        'variable': params.index,
        'coef': params.values,
        'ci_lower': conf[0],
        'ci_upper': conf[1]
    })
    
    if exclude_intercept:
        df_plot = df_plot[df_plot['variable'] != 'const']
    
    # Sort to control order of variables
    df_plot = df_plot.sort_values('coef')
    df_plot.reset_index(drop=True, inplace=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        df_plot['coef'],
        df_plot.index,
        xerr=[df_plot['coef'] - df_plot['ci_lower'], df_plot['ci_upper'] - df_plot['coef']],
        fmt='o', color='black', ecolor='gray', capsize=4
    )

    # Add horizontal line at x=0
    ax.axvline(x=0, color='red', linestyle='--')

    # Customize y-axis labels
    ax.set_yticks(df_plot.index)
    ax.set_yticklabels(df_plot['variable'])

    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.set_xlabel('Coefficient Estimate')
    ax.set_title(title)

    # Annotate coefficients just above each point
    if show_values:
        for i, row in df_plot.iterrows():
            ax.text(row['coef'], i + 0.25, f"{row['coef']:.2f}", ha='center', va='bottom', fontsize=9, color='blue')
