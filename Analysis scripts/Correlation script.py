import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import numpy as np
import re  # For sanitizing file names

# Function to sanitize file names for Windows compatibility
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Display settings
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.precision', 2)

# Load Excel file
file_path = 'Filtered data baseline SSP245.xlsx'
df = pd.read_excel(file_path)

#%% --- STEP 1: AGGREGATE ALL DATA TO MEANS PER GROUP --- 
print("Step 1: Calculating means per group...")
numeric_cols_to_agg = [
    'TMRR (%)', 'CMRR (%)', 'mean_baseline_treefrac', 'Treecover change (∆pp)',
    'mean_temp_baseline (K)', 'Degrees warming (∆K)', 'mean_baseline_precipitation', 'Precipitation change (∆mm/4 weeks)'
]
grouping_keys = ['Cluster_ID', 'Model', 'Scenario', 'Drying/Wetting', 'Century indication', 'Simulationtype']
agg_df = df.groupby(grouping_keys)[numeric_cols_to_agg].mean().reset_index()

# --- STEP 2: SPLIT AND MERGE BASELINE AND TARGET MEANS --- 
print("Step 2: Merging baseline and target means...")
baseline_means = agg_df[agg_df['Simulationtype'] == 'Baseline'].copy()
target_means = agg_df[agg_df['Simulationtype'] == 'Target'].copy()
merge_keys = ['Cluster_ID']
baseline_means_renamed = baseline_means.rename(columns={
    'TMRR (%)': 'TMRR_base', 'CMRR (%)': 'CMRR_base', 'mean_temp_baseline (K)': 'Temp_base',
    'mean_baseline_precipitation': 'Precip_base', 'mean_baseline_treefrac': 'Treefrac_base'
})
base_cols_for_merge = merge_keys + ['TMRR_base', 'CMRR_base', 'Temp_base', 'Precip_base', 'Treefrac_base']
analysis_df = pd.merge(target_means, baseline_means_renamed[base_cols_for_merge], on=merge_keys, how='left')

# --- STEP 3: CALCULATE BOTH ABSOLUTE AND RELATIVE CHANGES --- 
print("Step 3: Calculating all change variables (absolute and relative)...")
# Absolute changes in MRRs (in percentage points)
analysis_df['TMRR change (∆pp)'] = analysis_df['TMRR (%)'] - analysis_df['TMRR_base']
analysis_df['CMRR change (∆pp)'] = analysis_df['CMRR (%)'] - analysis_df['CMRR_base']

# Relative changes in MRRs
analysis_df['Relative TMRR change (∆%)'] = np.divide(analysis_df['TMRR change (∆pp)'], analysis_df['TMRR_base']) * 100
analysis_df['Relative CMRR change (∆%)'] = np.divide(analysis_df['CMRR change (∆pp)'], analysis_df['CMRR_base']) * 100

# Relative changes for independent variables
analysis_df['Relative Degrees warming (∆%)'] = np.divide(analysis_df['Degrees warming (∆K)'], analysis_df['Temp_base']) * 100
analysis_df['Relative Precipitation change (∆%)'] = np.divide(analysis_df['Precipitation change (∆mm/4 weeks)'], analysis_df['Precip_base']) * 100
analysis_df['Relative Treecover change (∆%)'] = np.divide(analysis_df['Treecover change (∆pp)'], analysis_df['Treefrac_base']) * 100

# --- STEP 4: DEFINE ANALYSIS FUNCTIONS TO AVOID REPETITION ---
def run_correlation_analysis(df, dep_vars, indep_vars):
    """Runs correlation analysis and saves results to a file."""
    print("\n--- Running Correlation Analysis ---")
    
    # Aggregate the final data
    all_vars_to_agg = dep_vars + indep_vars
    grouped_df = df.groupby(['Model', 'Scenario', 'Drying/Wetting', 'Century indication'])[all_vars_to_agg].mean().reset_index()

    drying_df = grouped_df[grouped_df['Drying/Wetting'] == 'drying']
    wetting_df = grouped_df[grouped_df['Drying/Wetting'] == 'wetting']
    correlation_results = []
    
    for group_name, group_df in [('Drying', drying_df), ('Wetting', wetting_df)]:
        for dep_var in dep_vars:
            for indep_var in indep_vars:
                valid_data = group_df[[dep_var, indep_var]].dropna()
                if len(valid_data) > 1 and len(valid_data[indep_var].unique()) > 1:
                    corr, p_val = pearsonr(valid_data[dep_var], valid_data[indep_var])
                    slope, _, _, _, _ = linregress(valid_data[indep_var], valid_data[dep_var])
                    correlation_results.append({'Group': group_name, 'Dependent': dep_var, 'Independent': indep_var, 'Correlation (r)': corr, 'P-value': p_val, 'Slope': slope})
                else:
                    correlation_results.append({'Group': group_name, 'Dependent': dep_var, 'Independent': indep_var, 'Correlation (r)': None, 'P-value': None, 'Slope': None})

    correlation_df = pd.DataFrame(correlation_results)
    correlation_df = correlation_df.sort_values(by=['Independent','Dependent', 'Group'], ascending=[True,False,True])
    filename = 'correlation_results.xlsx'
    print("\n🔹 Correlation Results Table:")
    
    print(correlation_df.to_string(index=False))
    correlation_df.to_excel(filename, index=False)
    print(f"\nCorrelation results saved to '{filename}'")
    return grouped_df

def run_plotting(df, dep_vars, indep_vars):
    """Generates scatter plots for the given analysis type."""
    print("\n--- Generating Plots ---")
    drying_df = df[df['Drying/Wetting'] == 'drying']
    wetting_df = df[df['Drying/Wetting'] == 'wetting']

    for indep_var in indep_vars:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)

        plot_configs = [
            {'ax': axes[0, 0], 'data': drying_df, 'dep_var': dep_vars[0], 'color': '#FF2424', 'title': 'Drying Group'},
            {'ax': axes[1, 0], 'data': drying_df, 'dep_var': dep_vars[1], 'color': '#FF2424', 'title': 'Drying Group'},
            {'ax': axes[0, 1], 'data': wetting_df, 'dep_var': dep_vars[0], 'color': '#0F58C4', 'title': 'Wetting Group'},
            {'ax': axes[1, 1], 'data': wetting_df, 'dep_var': dep_vars[1], 'color': '#0F58C4', 'title': 'Wetting Group'}
        ]

        for config in plot_configs:
            ax, data, dep_var, color, title = config.values()
            if ax == axes[0, 1]:
                sns.scatterplot(data=data, x=indep_var, y=dep_var, s=100, alpha=0.7, ax=ax, color='black', style='Model')
                ax.legend(
                    fontsize=14,
                    bbox_to_anchor=(1.02, 1),
                    loc='upper left',
                    borderaxespad=0
                )
            else:
                sns.scatterplot(data=data, x=indep_var, y=dep_var, s=100, alpha=0.7, ax=ax, color='black', style='Model', legend=False)
            valid_data = data[[indep_var, dep_var]].dropna()
            if len(valid_data) > 1 and len(valid_data[indep_var].unique()) > 1:
                sns.regplot(data=valid_data, x=indep_var, y=dep_var, scatter=False, color=color, ax=ax)
                slope, intercept, r, p, _ = linregress(valid_data[indep_var], valid_data[dep_var])
                # ax.text(0.05, 0.95, f'$y = {slope:.2f}x + {intercept:.2f}$\n$r = {r:.2f}$',
                #         transform=ax.transAxes, fontsize=16, verticalalignment='top')

            ax.set_title(title, fontsize=16)
            ax.set_xlabel(indep_var, fontsize=14)
            ax.set_ylabel(dep_var, fontsize=14)
            ax.tick_params(labelsize=12)
            # Add gridlines
            ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.7)

        plt.show()
        plt.savefig(sanitize_filename(f'Scatterplot_{indep_var}.png'), dpi=1200)


# --- STEP 5: RUN BOTH ABSOLUTE AND RELATIVE ANALYSES --- (Only absolute)

# 🔹 Scenario 1: Absolute Changes
print("\n" + "="*80 + "\nRUNNING ANALYSIS FOR ABSOLUTE CHANGES\n" + "="*80)
dep_vars_abs = ['TMRR change (∆pp)', 'CMRR change (∆pp)']
indep_vars_abs = ['Treecover change (∆pp)', 'Degrees warming (∆K)', 'Precipitation change (∆mm/4 weeks)']
absolute_results_df = run_correlation_analysis(analysis_df.copy(), dep_vars_abs, indep_vars_abs)
run_plotting(absolute_results_df, dep_vars_abs, indep_vars_abs)


# 🔹 Scenario 1: Relative Changes
# print("\n" + "="*80 + "\nRUNNING ANALYSIS FOR ABSOLUTE CHANGES\n" + "="*80)
# dep_vars_abs = ['Relative TMRR change (∆%)', 'Relative CMRR change (∆%)']
# indep_vars_abs = ['Relative Treecover change (∆%)', 'Relative Degrees warming (∆%)', 'Relative Precipitation change (∆%)']
# absolute_results_df = run_correlation_analysis(analysis_df.copy(), dep_vars_abs, indep_vars_abs)
# run_plotting(absolute_results_df, dep_vars_abs, indep_vars_abs)
