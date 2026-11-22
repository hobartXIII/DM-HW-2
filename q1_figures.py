"""Q1 報告用圖：速度與勝率、速度差分組、Fisher score"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import features

plt.rcParams.update({
    'font.family': ['Microsoft JhengHei', 'sans-serif'], 'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
    'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': .8, 'axes.axisbelow': True,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.dpi': 200, 'font.size': 10,
})
BLUE = '#2a78d6'
pk, cb = features.load()
X = features.build(pk, cb)
y = cb.y.to_numpy()

# 圖 1：每隻神奇寶貝的 Speed 與勝率
games = pd.concat([cb.First_pokemon, cb.Second_pokemon]).value_counts()
wr = (cb.Winner.value_counts().reindex(pk.index).fillna(0) / games.reindex(pk.index)).dropna()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, s in zip(axes, ['Speed', 'Attack', 'Defense']):
    r = np.corrcoef(pk.loc[wr.index, s], wr)[0, 1]
    ax.scatter(pk.loc[wr.index, s], wr, s=10, color=BLUE, alpha=.55, linewidths=0)
    ax.set_title(f'{s}（r = {r:.3f}）', fontsize=10)
    ax.set_xlabel(s)
axes[0].set_ylabel('個體勝率')
fig.tight_layout()
fig.savefig('fig_q1_stat_vs_winrate.png')

# 圖 2：速度差分組的先手勝率
bins = [-200, -50, -20, -5, -0.5, 0.5, 5, 20, 50, 200]
labels = ['≤-50', '-50~-20', '-20~-5', '-5~-1', '0（同速）', '1~5', '5~20', '20~50', '>50']
g = pd.Series(y).groupby(pd.cut(X.d_Speed, bins, labels=labels), observed=False).agg(['size', 'mean'])
fig, ax = plt.subplots(figsize=(8, 3.4))
bars = ax.bar(range(len(g)), g['mean'], width=.7, color=BLUE)
for i, (m, n) in enumerate(zip(g['mean'], g['size'])):
    ax.text(i, m + .02, f'{m:.1%}', ha='center', fontsize=8.5, color='#0b0b0b')
ax.set_xticks(range(len(g)), labels)
ax.set_ylim(0, 1.08)
ax.set_xlabel('Speed 差（先手 - 後手）')
ax.set_ylabel('先手勝率')
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
fig.tight_layout()
fig.savefig('fig_q1_speed_diff.png')


# 圖 3：候選特徵的 Fisher score
def fisher(x, y):
    mu = x.mean()
    num = sum((y == c).sum() * (x[y == c].mean() - mu) ** 2 for c in (0, 1))
    den = sum((y == c).sum() * x[y == c].var() for c in (0, 1))
    return num / den


C = X.copy()
P = features.build(pk, cb, 'prio')
for c in ['immune_1on2', 'immune_2on1', 'off_1on2', 'off_2on1']:
    C[c] = P[c]
C['d_Total'] = pk.loc[cb.First_pokemon, 'Total'].to_numpy() - pk.loc[cb.Second_pokemon, 'Total'].to_numpy()
C['d_Generation'] = pk.loc[cb.First_pokemon, 'Generation'].to_numpy() - pk.loc[cb.Second_pokemon, 'Generation'].to_numpy()
fs = pd.Series({c: fisher(C[c].to_numpy(float), y) for c in C.columns}).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(fs.index, fs.values, height=.65, color=BLUE)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
for i, v in enumerate(fs.values):
    ax.text(v * 1.08, i, f'{v:.4f}', va='center', fontsize=8.5)
ax.set_xlabel('Fisher score（對數刻度）')
ax.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig('fig_q1_fisher.png')
print(fs[::-1].round(4).to_string())
