"""Q1 探索式分析：找出影響對戰勝負的屬性"""
import sqlite3
import numpy as np
import pandas as pd

pd.set_option('display.width', 200)
con = sqlite3.connect('pokemon.db')
pk = pd.read_sql('select * from pokemon', con).set_index('#')
cb = pd.read_sql('select * from combats', con)
STATS = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

print('== 基本資訊 ==')
print(pk.dtypes.to_string())
print('缺值:\n', pk.isna().sum()[lambda s: s > 0].to_string())
print('Legendary 值:', pk['Legendary'].unique())
cb['y'] = (cb.Winner == cb.First_pokemon).astype(int)
print('combats:', len(cb), '先手勝比例', cb.y.mean().round(4))
print('自己打自己:', (cb.First_pokemon == cb.Second_pokemon).sum(),
      ' 重複對戰組合:', cb.duplicated(['First_pokemon', 'Second_pokemon']).sum())

# 每隻神奇寶貝的出場與勝率
games = pd.concat([cb.First_pokemon, cb.Second_pokemon]).value_counts()
wins = cb.Winner.value_counts()
pk['games'] = games.reindex(pk.index).fillna(0).astype(int)
pk['wins'] = wins.reindex(pk.index).fillna(0).astype(int)
pk['win_rate'] = pk.wins / pk.games.replace(0, np.nan)
print('從未出場:', (pk.games == 0).sum(), ' 出場次數 min/mean/max:', pk.games[pk.games > 0].min(), round(pk.games.mean(), 1), pk.games.max())

print('\n== 數值屬性平均、標準差 ==')
pk['Total'] = pk[STATS].sum(axis=1)
print(pk[STATS + ['Total']].agg(['mean', 'std', 'min', 'max']).round(2).to_string())

print('\n== 數值屬性與個體勝率的 Pearson 相關 ==')
print(pk[STATS + ['Total', 'Generation']].corrwith(pk.win_rate).round(3).sort_values(ascending=False).to_string())

print('\n== 勝率前 / 後五名 ==')
cols = ['Name', 'Type 1', 'Type 2'] + STATS + ['Legendary', 'games', 'win_rate']
print(pk.sort_values('win_rate', ascending=False)[cols].head(5).round(3).to_string())
print(pk.sort_values('win_rate')[cols].head(5).round(3).to_string())

print('\n== Type 1 分布與平均勝率 ==')
t = pk.groupby('Type 1').agg(n=('Name', 'size'), win=('win_rate', 'mean'), speed=('Speed', 'mean'), total=('Total', 'mean'))
print(t.sort_values('win', ascending=False).round(3).to_string())

print('\n== 單屬性 vs 雙屬性 ==')
pk['dual'] = pk['Type 2'].notna()
print(pk.groupby('dual').agg(n=('Name', 'size'), win=('win_rate', 'mean'), total=('Total', 'mean')).round(3).to_string())

print('\n== Legendary / Generation ==')
print(pk.groupby('Legendary').agg(n=('Name', 'size'), win=('win_rate', 'mean'), total=('Total', 'mean')).round(3).to_string())
print(pk.groupby('Generation').agg(n=('Name', 'size'), win=('win_rate', 'mean')).round(3).to_string())

print('\n== 對戰層級：差值 (First - Second) 的符號與先手勝率 ==')
A = pk.loc[cb.First_pokemon, STATS + ['Total']].to_numpy(float)
B = pk.loc[cb.Second_pokemon, STATS + ['Total']].to_numpy(float)
D = pd.DataFrame(A - B, columns=[f'd_{s}' for s in STATS + ['Total']])
rows = []
for c in D.columns:
    d = D[c]
    rows.append({'feature': c,
                 'P(first win | d>0)': cb.y[d > 0].mean(),
                 'P(first win | d<0)': cb.y[d < 0].mean(),
                 'ties': int((d == 0).sum()),
                 'corr(d, y)': np.corrcoef(d, cb.y)[0, 1]})
print(pd.DataFrame(rows).round(4).to_string(index=False))


def fisher(x, y):
    """課堂 W10 的 Fisher score：類間距離 / 類內變異"""
    mu = x.mean()
    num = sum((y == c).sum() * (x[y == c].mean() - mu) ** 2 for c in (0, 1))
    den = sum((y == c).sum() * x[y == c].var(ddof=0) for c in (0, 1))
    return num / den


print('\n== 候選特徵的 Fisher score（對戰層級）==')
cand = D.copy()
lg = pk.Legendary.astype(str).str.upper().eq('TRUE').astype(int)
cand['first_faster'] = (D.d_Speed > 0).astype(int)
cand['d_Legendary'] = lg.loc[cb.First_pokemon].to_numpy() - lg.loc[cb.Second_pokemon].to_numpy()
cand['d_Generation'] = pk.loc[cb.First_pokemon, 'Generation'].to_numpy() - pk.loc[cb.Second_pokemon, 'Generation'].to_numpy()
cand['ratio_Speed'] = A[:, 5] / B[:, 5]
cand['d_Atk_vs_Def'] = (A[:, 1] - B[:, 2]) - (B[:, 1] - A[:, 2])
cand['d_SpA_vs_SpD'] = (A[:, 3] - B[:, 4]) - (B[:, 3] - A[:, 4])
fs = {c: fisher(cand[c].to_numpy(float), cb.y.to_numpy()) for c in cand.columns}
print(pd.Series(fs).sort_values(ascending=False).round(4).to_string())

print('\n== 較快的一方勝率，依速度差分組 ==')
bins = [-200, -50, -20, -5, -0.5, 0.5, 5, 20, 50, 200]
g = cb.y.groupby(pd.cut(D.d_Speed, bins)).agg(['size', 'mean'])
print(g.round(3).to_string())

print('\n== Type 1 對戰（以先手屬性分組的先手勝率，扣掉速度影響前）==')
tf = pk.loc[cb.First_pokemon, 'Type 1'].to_numpy()
print(cb.y.groupby(tf).agg(['size', 'mean']).sort_values('mean', ascending=False).round(3).to_string())

print('\n== 屬性相剋（features.py）：控制速度後是否還有影響 ==')
import features
pk2, cb2 = features.load()
X = features.build(pk2, cb2)
X['eff_adv'] = X.eff_1on2 - X.eff_2on1
X['y'] = cb2.y
X['speed'] = np.where(X.first_faster == 1, 'first faster', 'first slower/tie')
X['adv'] = np.sign(X.eff_adv).map({-1: 'disadv', 0: 'neutral', 1: 'adv'})
print(X.pivot_table(index='adv', columns='speed', values='y', aggfunc=['size', 'mean']).round(3).to_string())
print('Fisher eff_1on2 %.4f  eff_2on1 %.4f  eff_adv %.4f' % tuple(fisher(X[c].to_numpy(float), X.y.to_numpy()) for c in ['eff_1on2', 'eff_2on1', 'eff_adv']))

print('\n== 傳說神奇寶貝：控制速度後 ==')
X['leg'] = X.d_Legendary.map({-1: 'second legend', 0: 'same', 1: 'first legend'})
print(X.pivot_table(index='leg', columns='speed', values='y', aggfunc=['size', 'mean']).round(3).to_string())

print('\n== 單/雙屬性：控制速度後（先手是否雙屬性）==')
X['dual1'] = pk2.loc[cb2.First_pokemon, 'Type 2'].notna().to_numpy()
print(X.pivot_table(index='dual1', columns='speed', values='y', aggfunc=['size', 'mean']).round(3).to_string())

print('\n== 只用 first_faster 一條規則的準確率 ==', round((X.first_faster == X.y).mean(), 4))
