import pandas as pd
import matplotlib.pyplot as plt

# 1. Define Updated Data
raw_data = [
    # vLLM, ISL=1000
    {"Engine": "vLLM", "ISL": "1000", "C": "2", "Kernel": "GEMM_64x64_stage_64x5", "Pct": 54.3},
    {"Engine": "vLLM", "ISL": "1000", "C": "2", "Kernel": "GEMM_64x64_stage_64x6", "Pct": 28.0},
    {"Engine": "vLLM", "ISL": "1000", "C": "2", "Kernel": "ATTN (kernel_unified_uniform)", "Pct": 7.4},
    {"Engine": "vLLM", "ISL": "1000", "C": "32", "Kernel": "GEMM_254x64_stage_64x3", "Pct": 49.2},
    {"Engine": "vLLM", "ISL": "1000", "C": "32", "Kernel": "ATTN (kernel_unified_uniform)", "Pct": 33.5},
    {"Engine": "vLLM", "ISL": "1000", "C": "32", "Kernel": "GEMM_64x64_stage_64x6", "Pct": 10.5},
    
    # Sglang, ISL=1000 (UPDATED DATA)
    {"Engine": "Sglang", "ISL": "1000", "C": "2", "Kernel": "GEMM_64x64_stage_64x5", "Pct": 56.7},
    {"Engine": "Sglang", "ISL": "1000", "C": "2", "Kernel": "GEMM_64x64_stage_64x6", "Pct": 28.8},
    {"Engine": "Sglang", "ISL": "1000", "C": "2", "Kernel": "ATTN (fwd_grouped_kernel_stage1)", "Pct": 4.6},
    {"Engine": "Sglang", "ISL": "1000", "C": "32", "Kernel": "GEMM_254x64_stage_64x3", "Pct": 55.5},
    {"Engine": "Sglang", "ISL": "1000", "C": "32", "Kernel": "ATTN (fwd_grouped_kernel_stage1)", "Pct": 24.1},
    {"Engine": "Sglang", "ISL": "1000", "C": "32", "Kernel": "GEMM_64x64_stage_64x6", "Pct": 12.3},
    
    # vLLM, ISL=10k
    {"Engine": "vLLM", "ISL": "10k", "C": "2", "Kernel": "GEMM_64x64_stage_64x5", "Pct": 44.1},
    {"Engine": "vLLM", "ISL": "10k", "C": "2", "Kernel": "GEMM_64x64_stage_64x6", "Pct": 22.9},
    {"Engine": "vLLM", "ISL": "10k", "C": "2", "Kernel": "ATTN (kernel_unified_uniform)", "Pct": 22.4},
    {"Engine": "vLLM", "ISL": "10k", "C": "32", "Kernel": "ATTN (kernel_unified_uniform)", "Pct": 71.6},
    {"Engine": "vLLM", "ISL": "10k", "C": "32", "Kernel": "GEMM_254x64_stage_64x3", "Pct": 20.8},
    {"Engine": "vLLM", "ISL": "10k", "C": "32", "Kernel": "GEMM_64x64_stage_64x6", "Pct": 4.5},
    
    # Sglang, ISL=10k
    {"Engine": "Sglang", "ISL": "10k", "C": "2", "Kernel": "GEMM_64x64_stage_64x5", "Pct": 45.7},
    {"Engine": "Sglang", "ISL": "10k", "C": "2", "Kernel": "GEMM_64x64_stage_64x6", "Pct": 23.5},
    {"Engine": "Sglang", "ISL": "10k", "C": "2", "Kernel": "ATTN (fwd_grouped_kernel_stage1)", "Pct": 22.6},
    {"Engine": "Sglang", "ISL": "10k", "C": "32", "Kernel": "ATTN (fwd_grouped_kernel_stage1)", "Pct": 68.4},
    {"Engine": "Sglang", "ISL": "10k", "C": "32", "Kernel": "GEMM_254x64_stage_64x3", "Pct": 22.9},
    {"Engine": "Sglang", "ISL": "10k", "C": "32", "Kernel": "GEMM_64x64_stage_64x6", "Pct": 5.0}
]

df = pd.DataFrame(raw_data)

# To create a clean facet grid with stacked bars, we can iterate over subplots manually for maximum control.
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
engines = ['vLLM', 'Sglang']
isls = ['1000', '10k']

colors = {
    'ATTN (fwd_grouped_kernel_stage1)': '#ff7f0e',       # Dark Orange
    'ATTN (kernel_unified_uniform)': '#ffbb78',          # Light Orange
    'GEMM_254x64_stage_64x3': '#1f77b4',                 # Dark Blue
    'GEMM_64x64_stage_64x5': '#2ca02c',                  # Green
    'GEMM_64x64_stage_64x6': '#98df8a',                  # Light Green
    'Others': '#d3d3d3'                                  # Light Gray
}

for i, isl in enumerate(isls):
    for j, engine in enumerate(engines):
        ax = axes[i, j]
        sub_df = df[(df['Engine'] == engine) & (df['ISL'] == isl)]
        
        # Pivot specifically for this facet
        pivot_sub = sub_df.pivot_table(index='C', columns='Kernel', values='Pct', fill_value=0)
        pivot_sub['Others'] = 100 - pivot_sub.sum(axis=1)
        
        # Reorder columns to match main color mapping
        cols = [c for c in colors if c in pivot_sub.columns]
        pivot_sub = pivot_sub[cols]
        
        # Plot
        color_sub = [colors[c] for c in pivot_sub.columns]
        pivot_sub.plot(kind='barh', stacked=True, ax=ax, color=color_sub, edgecolor='white', legend=False)
        
        ax.set_title(f"{engine} | ISL = {isl}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Concurrency (C)")
        ax.set_xlabel("Execution Time (%)" if i == 1 else "")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        
        # Add bar labels
        for container in ax.containers:
            labels = [f'{v:.1f}%' if v > 5.0 else '' for v in container.datavalues]
            ax.bar_label(container, labels=labels, label_type='center', color='black', weight='bold', fontsize=8.5)

# Handle master legend
handles, labels = axes[0, 0].get_legend_handles_labels()
# Gather handles from all plots to make sure we capture all kernels
all_handles = {}
for row in axes:
    for ax in row:
        h, l = ax.get_legend_handles_labels()
        for han, lab in zip(h, l):
            all_handles[lab] = han

fig.legend(all_handles.values(), all_handles.keys(), title="Kernel Operation Category", 
           bbox_to_anchor=(0.5, -0.02), loc='upper center', ncol=3, fontsize=10, title_fontsize=11)

plt.suptitle("Workload Shift Analysis: Kernel Swap & Attention Bottlenecking", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('facet_kernel_analysis.png', dpi=300, bbox_inches='tight')
print("Facet plot generated successfully.")