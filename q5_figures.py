"""Q5 報告用圖：錯誤率 vs 速度差、vs 屬性優劣（先手較快的對戰）"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': ['Microsoft JhengHei', 'sans-serif'], 'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
    'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e4e3df',
    'axes.axisbelow': True, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.dpi': 200,
})
BLUE = '#2a78d6'
D = pd.read_csv('q5_test_predictions.csv')
pct = matplotlib.ticker.PercentFormatter(1, decimals=0)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
sub = D[D.speed_rel != '同速']
labels = ['1~5', '6~20', '21~50', '>50']
g = (1 - sub.ok).groupby(pd.cut(sub.d_Speed.abs(), [0, 5, 20, 50, 200], labels=labels), observed=False).mean()
axes[0].bar(labels, g.values, width=.65, color=BLUE)
axes[0].set_title('速度差越大，模型錯越多', fontsize=11)
axes[0].set_xlabel('|Speed 差|')
axes[0].set_ylabel('SVM 錯誤率')

fast = D[D.speed_rel == '先手較快']
order = ['屬性劣勢', '屬性平手', '屬性優勢']
g2 = (1 - fast.ok).groupby(fast.adv).mean().reindex(order)
axes[1].bar(order, g2.values, width=.65, color=BLUE)
axes[1].set_title('比較快的一方：被剋時最容易判錯', fontsize=11)
axes[1].set_ylabel('SVM 錯誤率')

pb = pd.cut(D.p, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True,
            labels=['0~.1', '.1~.3', '.3~.5', '.5~.7', '.7~.9', '.9~1'])
cnt = (1 - D.ok).groupby(pb, observed=False).sum()
axes[2].bar(cnt.index.astype(str), cnt.values, width=.65, color=BLUE)
axes[2].set_title('錯誤大多是「很有把握的錯」', fontsize=11)
axes[2].set_xlabel('P(先手勝)')
axes[2].set_ylabel('錯誤筆數')
for ax, vals, f in [(axes[0], g.values, '{:.1%}'), (axes[1], g2.values, '{:.1%}'), (axes[2], cnt.values, '{:.0f}')]:
    for i, v in enumerate(vals):
        ax.text(i, v, f.format(v), ha='center', va='bottom', fontsize=8.5)
for ax in axes[:2]:
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0, max(ax.get_ylim()[1], .13))
fig.tight_layout()
fig.savefig('fig_q5_errors.png')
