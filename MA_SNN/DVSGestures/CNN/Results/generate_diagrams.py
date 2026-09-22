import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'Greedy': '#2196F3',
    'Self-Consistency': '#4CAF50',
    'Best-of-N': '#FF9800',
    'Router': '#9C27B0',
    'Tree Search': '#F44336',
}

STRATEGIES = ['Greedy', 'Self-Consistency', 'Best-of-N', 'Router', 'Tree Search']
FILES = {
    'Greedy': 'greedy_qwen1_5b_mbpp_test_limit257.csv',
    'Self-Consistency': 'self_consistency_qwen1_5b_mbpp_test_limit257.csv',
    'Best-of-N': 'best_of_n_qwen1_5b_mbpp_test_limit257.csv',
    'Router': 'router_qwen1_5b_mbpp_test_limit257.csv',
    'Tree Search': 'tree_search_qwen1_5b_mbpp_test_limit257.csv',
}

data = {}
for name, fname in FILES.items():
    df = pd.read_csv(fname)
    data[name] = df

# Compute summary stats
summary = {}
for name in STRATEGIES:
    df = data[name]
    total = len(df)
    passed = (df['passed'] == True).sum() + (df['passed'] == 'True').sum()
    failed = total - passed
    avg_flops = df['flops_estimated_flops'].mean()
    passed_mask = (df['passed'] == True) | (df['passed'] == 'True')
    failed_mask = ~passed_mask
    avg_flops_passed = df.loc[passed_mask, 'flops_estimated_flops'].mean()
    avg_flops_failed = df.loc[failed_mask, 'flops_estimated_flops'].mean()
    total_flops = df['flops_estimated_flops'].sum()
    
    errors = df[df['error'].notna() & (df['error'] != '')]
    
    name_err = errors['error'].str.contains('NameError', na=False).sum()
    type_err = errors['error'].str.contains('TypeError', na=False).sum()
    index_err = errors['error'].str.contains('IndexError', na=False).sum()
    assert_err = errors['error'].str.contains('AssertionError', na=False).sum()
    syntax_err = errors['error'].str.contains('SyntaxError', na=False).sum()
    timeout_err = errors['error'].str.contains('TimeoutExpired', na=False).sum()
    other_err = len(errors) - name_err - type_err - index_err - assert_err - syntax_err - timeout_err
    
    summary[name] = {
        'total': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': passed / total * 100,
        'avg_flops': avg_flops,
        'avg_flops_passed': avg_flops_passed,
        'avg_flops_failed': avg_flops_failed,
        'total_flops': total_flops,
        'errors': {
            'AssertionError': int(assert_err),
            'NameError': int(name_err),
            'TypeError': int(type_err),
            'IndexError': int(index_err),
            'SyntaxError': int(syntax_err),
            'Timeout': int(timeout_err),
            'Other': max(0, int(other_err)),
        }
    }


# ============================================================
# Diagram 1: Accuracy Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
rates = [summary[s]['pass_rate'] for s in STRATEGIES]
colors = [COLORS[s] for s in STRATEGIES]
bars = ax.bar(STRATEGIES, rates, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Pass@1 Accuracy (%)')
ax.set_title('Pass@1 Accuracy Across Decoding Strategies (MBPP, Qwen-1.5B)')
ax.set_ylim(0, 100)
ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig('diagram1_accuracy_comparison.png')
plt.close()
print("Diagram 1 saved: diagram1_accuracy_comparison.png")


# ============================================================
# Diagram 2: Cost-Accuracy Scatter Plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.5))
for s in STRATEGIES:
    ax.scatter(
        summary[s]['avg_flops'] / 1e9,
        summary[s]['pass_rate'],
        color=COLORS[s], s=200, edgecolors='black', linewidth=0.8, zorder=5, label=s
    )
    ax.annotate(s, (summary[s]['avg_flops'] / 1e9, summary[s]['pass_rate']),
                textcoords="offset points", xytext=(8, 8), fontsize=10)
ax.set_xlabel('Average Estimated FLOPs per Problem (Billions)')
ax.set_ylabel('Pass@1 Accuracy (%)')
ax.set_title('Computational Cost vs. Accuracy Trade-off')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
ax.set_ylim(40, 85)
plt.tight_layout()
plt.savefig('diagram2_cost_accuracy_scatter.png')
plt.close()
print("Diagram 2 saved: diagram2_cost_accuracy_scatter.png")


# ============================================================
# Diagram 3: Total Cost vs Accuracy Grouped Bar
# ============================================================
fig, ax1 = plt.subplots(figsize=(9, 5))
x = np.arange(len(STRATEGIES))
width = 0.35

total_flops_tb = [summary[s]['total_flops'] / 1e12 for s in STRATEGIES]
rates = [summary[s]['pass_rate'] for s in STRATEGIES]

bars1 = ax1.bar(x - width/2, total_flops_tb, width, label='Total FLOPs (TFLOPs)',
                color=[COLORS[s] for s in STRATEGIES], edgecolor='black', linewidth=0.5, alpha=0.85)
ax1.set_ylabel('Total Estimated FLOPs (TFLOPs)', color='black')
ax1.set_xlabel('Decoding Strategy')
ax1.set_xticks(x)
ax1.set_xticklabels(STRATEGIES, rotation=15, ha='right')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, rates, width, label='Pass@1 Accuracy (%)',
                color=[COLORS[s] for s in STRATEGIES], edgecolor='black', linewidth=0.5, alpha=0.5, hatch='//')
ax2.set_ylabel('Pass@1 Accuracy (%)', color='black')
ax2.set_ylim(0, 100)

for bar, val in zip(bars2, rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

ax1.set_title('Total Computational Cost vs. Pass@1 Accuracy')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('diagram3_cost_vs_accuracy_bars.png')
plt.close()
print("Diagram 3 saved: diagram3_cost_vs_accuracy_bars.png")


# ============================================================
# Diagram 4: Error Type Distribution (Stacked Bar)
# ============================================================
error_types = ['AssertionError', 'NameError', 'TypeError', 'IndexError', 'SyntaxError', 'Other']
error_colors = ['#E53935', '#FB8C00', '#FDD835', '#66BB6A', '#42A5F5', '#BDBDBD']

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(STRATEGIES))
width = 0.55
bottom = np.zeros(len(STRATEGIES))

for i, etype in enumerate(error_types):
    vals = [summary[s]['errors'].get(etype, 0) for s in STRATEGIES]
    bars = ax.bar(x, vals, width, bottom=bottom, label=etype, color=error_colors[i],
                  edgecolor='black', linewidth=0.3)
    bottom += np.array(vals)

totals = [sum(summary[s]['errors'].values()) for s in STRATEGIES]
for i, (t, b) in enumerate(zip(totals, x)):
    if t > 0:
        ax.text(b, t + 0.5, str(t), ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(STRATEGIES, rotation=15, ha='right')
ax.set_ylabel('Number of Failed Problems')
ax.set_title('Error Type Distribution Across Strategies')
ax.legend(loc='upper right', fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig('diagram4_error_distribution.png')
plt.close()
print("Diagram 4 saved: diagram4_error_distribution.png")


# ============================================================
# Diagram 5: Cost Breakdown Passed vs Failed
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(STRATEGIES))
width = 0.32

avg_passed = [summary[s]['avg_flops_passed'] / 1e9 for s in STRATEGIES]
avg_failed = [summary[s]['avg_flops_failed'] / 1e9 for s in STRATEGIES]

bars_p = ax.bar(x - width/2, avg_passed, width, label='Passed Problems',
                color=[COLORS[s] for s in STRATEGIES], edgecolor='black', linewidth=0.5, alpha=0.85)
bars_f = ax.bar(x + width/2, avg_failed, width, label='Failed Problems',
                color=[COLORS[s] for s in STRATEGIES], edgecolor='black', linewidth=0.5, alpha=0.45, hatch='xx')

for bar, val in zip(bars_p, avg_passed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars_f, avg_failed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(STRATEGIES, rotation=15, ha='right')
ax.set_ylabel('Average Estimated FLOPs per Problem (Billions)')
ax.set_title('Average Computational Cost: Passed vs. Failed Problems')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('diagram5_cost_passed_vs_failed.png')
plt.close()
print("Diagram 5 saved: diagram5_cost_passed_vs_failed.png")


# ============================================================
# Diagram 6: Router Routing Distribution (Pie Chart)
# ============================================================
router_df = data['Router']
if 'routed_to' in router_df.columns:
    routing = router_df['routed_to'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    route_labels = routing.index.tolist()
    route_values = routing.values
    route_colors = ['#FF9800', '#F44336'] if len(route_labels) == 2 else plt.cm.Set2(np.linspace(0, 1, len(route_labels)))
    
    wedges, texts, autotexts = ax.pie(
        route_values, labels=route_labels, autopct='%1.1f%%',
        colors=route_colors, startangle=90,
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8}
    )
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_title('Router Strategy: Problem Routing Distribution')
    plt.tight_layout()
    plt.savefig('diagram6_router_distribution.png')
    plt.close()
    print("Diagram 6 saved: diagram6_router_distribution.png")
else:
    print("Diagram 6 skipped: 'routed_to' column not found in router data")


# ============================================================
# Bonus: Per-Problem Pass/Fail Heatmap
# ============================================================
problem_ids = sorted(set(data['Greedy']['problem_id']))
heatmap_data = np.zeros((len(STRATEGIES), len(problem_ids)))
for j, pid in enumerate(problem_ids):
    for i, s in enumerate(STRATEGIES):
        df = data[s]
        row = df[df['problem_id'] == pid]
        if len(row) > 0:
            val = row.iloc[0]['passed']
            heatmap_data[i, j] = 1 if (val == True or val == 'True') else 0

fig, ax = plt.subplots(figsize=(16, 3.5))
cmap = plt.cm.colors.ListedColormap(['#F44336', '#4CAF50'])
ax.imshow(heatmap_data, aspect='auto', cmap=cmap, interpolation='nearest')
ax.set_yticks(range(len(STRATEGIES)))
ax.set_yticklabels(STRATEGIES, fontsize=10)
ax.set_xlabel('Problem ID (sorted)')
ax.set_title('Per-Problem Pass/Fail Heatmap (Green=Pass, Red=Fail)')
ax.set_xticks([])
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', edgecolor='black', label='Pass'),
                   Patch(facecolor='#F44336', edgecolor='black', label='Fail')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('diagram7_per_problem_heatmap.png')
plt.close()
print("Diagram 7 saved: diagram7_per_problem_heatmap.png")


print("\nAll diagrams generated successfully!")
