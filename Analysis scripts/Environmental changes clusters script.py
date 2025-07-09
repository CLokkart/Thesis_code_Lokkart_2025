import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# PART 1: SCRIPT FOR CREATING THE EXCEL FILE CONTAINING MEAN CHANGES
# ==============================================================================

#% DETERMINE THE RANGES FOR THE ENVIRONMENTAL VARIABLES
print("\n📊 model Means for Selected Independent Variables:")
vars_for_tables = ["mean_significant_treefrac_change (%)", 'mean_significant_temp_change (K)', 'mean_significant_pr_change']

# Ensure the Excel file is in the same directory or provide a full path
df_clusters = pd.read_excel('cluster_data.xlsx')

# Remove clusters that are too small
df_clusters = df_clusters[~(
    ((df_clusters['model'] == 'EC-Earth3') & (df_clusters['number_of_gridcells'] < 119)) |
    ((df_clusters['model'] == 'MPI-ESM1-2-HR') & (df_clusters['number_of_gridcells'] < 67))
)].reset_index(drop=True)

# The column name in the excel file is 'Drying/Wetting', which is used here.
century_stats_df = df_clusters.groupby(['model', 'scenario', 'target_period', 'type'])[vars_for_tables].mean().reset_index()

# --- Define mapping for century column names ---
century_mapping = {'(2050, 2059)': 'Mid-Century',
                   '(2090, 2099)': 'End-Century'}

# Rename the target_period column *before* the loop for easier use in plotting
century_stats_df.rename(columns={'target_period': 'Time Period'}, inplace=True)
century_stats_df['Time Period'] = century_stats_df['Time Period'].replace(century_mapping)
century_stats_df.rename(columns={'model': 'Model'}, inplace=True)
century_stats_df.rename(columns={'type': 'Drying/Wetting'}, inplace=True)

# Format scenario names
ssp_mapping = {
    'ssp126': 'SSP1-2.6',
    'ssp245': 'SSP2-4.5',
    'ssp370': 'SSP3-7.0',
    'ssp585': 'SSP5-8.5'
}
century_stats_df['scenario'] = century_stats_df['scenario'].replace(ssp_mapping)

# Open a single ExcelWriter context for all sheets
with pd.ExcelWriter('Environmental variable changes (Treefrac and Temp).xlsx') as writer:
    for var_to_summarize in vars_for_tables:
        print(f"\n--- Summary Table (model Means) for: {var_to_summarize} ---")
        
        mean_col_for_var = f"{var_to_summarize}"
        
        # Subset relevant columns
        subset_df = century_stats_df[['Model', 'scenario', 'Time Period', 'Drying/Wetting', mean_col_for_var]]
        
        # Pivot so type becomes columns
        table = subset_df.pivot_table(
            index=['Model', 'scenario', 'Time Period'],
            columns='Drying/Wetting',
            values=mean_col_for_var
        ).reset_index()
        
        # The values in the 'Drying/Wetting' column are likely 'drying' and 'wetting' in lowercase.
        table.rename(columns={
            'drying': f"{var_to_summarize} (Drying)",
            'wetting': f"{var_to_summarize} (Wetting)"
        }, inplace=True)

        # Reorder columns for clarity
        # We need to handle potential KeyErrors if 'drying' or 'wetting' columns don't exist after pivot
        final_cols = ['Model', 'scenario', 'Time Period']
        if f"{var_to_summarize} (Drying)" in table.columns:
            final_cols.append(f"{var_to_summarize} (Drying)")
        if f"{var_to_summarize} (Wetting)" in table.columns:
            final_cols.append(f"{var_to_summarize} (Wetting)")

        table = table[final_cols]
        
        # Make 'Time Period' a categorical column for sorting
        table['Time Period'] = pd.Categorical(table['Time Period'],
                                              categories=['Mid-Century', 'End-Century'],
                                              ordered=True)

        # Sort and set index
        table = table.sort_values(by=['scenario', 'Time Period', 'Model'])
        table = table.set_index(['scenario', 'Time Period', 'Model'])

        # Save each table to its own sheet using the first word of the variable name
        sheet_name = var_to_summarize.split(' ')[0]
        table.to_excel(writer, sheet_name=sheet_name)
        
        print(table)

print("\n✅ Excel file 'Environmental variable changes (Treefrac and Temp).xlsx' has been created successfully.")


# ==============================================================================
# PART 2: SECTION FOR CREATING CATEGORICAL PLOTS 
# ==============================================================================
print("\n🎨 Creating categorical plots to visualize data spread with manual jitter...")

# Step 1: Create combined category column
century_stats_df['Scenario_TimePeriod'] = century_stats_df.apply(
    lambda row: f"{row['scenario']} ({row['Time Period']})", axis=1
)

# Step 2: Define desired order: each SSP followed by Mid-Century then End-Century
ssp_order = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
time_order = ['Mid-Century', 'End-Century']

# Create ordered list of combined categories that actually exist in the data
category_order = [
    f"{ssp} ({tp})"
    for ssp in ssp_order
    for tp in time_order
    if f"{ssp} ({tp})" in century_stats_df['Scenario_TimePeriod'].unique()
]

cat_to_num = {cat: i for i, cat in enumerate(category_order)}
num_to_cat = {v: k for k, v in cat_to_num.items()}

century_stats_df['x_numeric'] = century_stats_df['Scenario_TimePeriod'].map(cat_to_num)
# np.random.seed(50)
# century_stats_df['x_jittered'] = century_stats_df['x_numeric'] + np.random.uniform(-0.20, 0.20, size=len(century_stats_df))

# Step 3: Assign fixed offsets per model
unique_models = sorted(century_stats_df['Model'].unique())
offset_step = 0.15  # adjust this for tighter or wider spacing
model_offsets = {model: ((i+0.5) - len(unique_models) / 2) * offset_step for i, model in enumerate(unique_models)}

# Step 4: Apply the offsets to get the jittered (now structured) x-axis
century_stats_df['x_jittered'] = century_stats_df.apply(
    lambda row: row['x_numeric'] + model_offsets[row['Model']],
    axis=1
)

# Custom colors
custom_palette = {"drying": "#FF2424", "wetting": "#0F58C4"}

def plot_jittered(df, y_col, y_label, filename):
    plt.figure(figsize=(14, 6))
    ax = sns.scatterplot(
        data=df,
        x='x_jittered',
        y=y_col,
        hue='Drying/Wetting',
        style='Model',
        palette=custom_palette,
        s=100,
        alpha=0.85
    )
    # Set x-ticks back to category labels
    ax.set_xticks(list(cat_to_num.values()))
    ax.set_xticklabels([num_to_cat[i] for i in sorted(num_to_cat.keys())], rotation=45, ha='right', fontsize=12)
    
    # Add vertical separator lines between categories
    for i in range(len(cat_to_num) - 1):
        x_pos = i + 0.5  # halfway between categories
        ax.axvline(x=x_pos, color='gray', linestyle='-', linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=14)
    ax.axhline(0, color='black', linewidth=1)
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)  # Optional styling
    # Move legend outside the plot area (right side)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0,
        fontsize=12,         # regular legend labels font size
        labelspacing=1
    )
    
    plt.tight_layout()
    plt.savefig(filename, dpi=1200)
    print(f"    ✅ Plot saved as '{filename}'")

# Plot 1: Tree Fraction Change
print("    - Generating plot for Tree Fraction Change...")
plot_jittered(
    df=century_stats_df,
    y_col='mean_significant_treefrac_change (%)',
    y_label="Tree cover Fraction Change (∆pp)",
    filename='treefrac_change_by_scenario_time_period_combined_x.png'
)

# Plot 2: Temperature Change
print("    - Generating plot for Temperature Change...")
plot_jittered(
    df=century_stats_df,
    y_col='mean_significant_temp_change (K)',
    y_label="Degrees Warming (∆K)",
    filename='temp_change_by_scenario_time_period_combined_x.png'
)

# Plot 3: Precipitation Change
print("    - Generating plot for Precipitation Change...")
plot_jittered(
    df=century_stats_df,
    y_col='mean_significant_pr_change',
    y_label="Precipitation Change\n(∆mm/4 weeks)",
    filename='prec_change_by_scenario_time_period_combined_x.png'
)

plt.show()