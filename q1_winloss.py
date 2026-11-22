"""Q1 補充：勝方 / 敗方各數值屬性的平均與標準差、單 / 雙屬性比較（含控制速度）、Legendary、Generation"""
import numpy as np
import pandas as pd
import features as F

pk, cb = F.load()
S = F.STATS + ['Total']

# 1. 每場對戰的勝方與敗方
W = pk.loc[cb.Winner].reset_index(drop=True)
L = pk.loc[np.where(cb.Winner == cb.First_pokemon, cb.Second_pokemon, cb.First_pokemon)].reset_index(drop=True)
print('== 勝方 / 敗方 數值屬性（50,000 場）==')
rows = [{'stat': s, 'win_mean': W[s].mean(), 'win_std': W[s].std(), 'lose_mean': L[s].mean(), 'lose_std': L[s].std(),
         'diff': W[s].mean() - L[s].mean(), 'win_higher': (W[s] > L[s]).mean()} for s in S]
print(pd.DataFrame(rows).round(3).to_string(index=False))
print('Legendary 比例  勝方', round(W.Legendary.mean(), 4), ' 敗方', round(L.Legendary.mean(), 4))
print('Generation 平均 勝方', round(W.Generation.mean(), 3), ' 敗方', round(L.Generation.mean(), 3))

# 2. 個體勝率
games = cb.First_pokemon.value_counts().add(cb.Second_pokemon.value_counts(), fill_value=0)
pk['wr'] = cb.Winner.value_counts().reindex(pk.index).fillna(0) / games.reindex(pk.index)
pk['dual'] = pk['Type 2'].notna()
agg = dict(n=('wr', 'size'), win_rate=('wr', 'mean'), speed=('Speed', 'mean'), total=('Total', 'mean'))
print('\n== 單 / 雙屬性 ==')
print(pk.groupby('dual').agg(**agg).round(3).to_string())
print('\n== 單 / 雙屬性：依 Speed 分組後的平均勝率 ==')
pk['speed_bin'] = pd.cut(pk.Speed, [0, 50, 70, 90, 200], labels=['≤50', '51~70', '71~90', '>90'])
print(pk.groupby(['speed_bin', 'dual'], observed=True).wr.agg(['size', 'mean']).round(3).unstack().to_string())
print('\n== Legendary ==')
print(pk.groupby('Legendary').agg(**agg).round(3).to_string())
print('\n== Generation ==')
print(pk.groupby('Generation').agg(**agg).round(3).to_string())

# 傳說 / 600族（準神）/ 一般神奇寶貝
PSEUDO = ['Dragonite', 'Tyranitar', 'Salamence', 'Metagross', 'Garchomp', 'Hydreigon', 'Goodra']
name = pk.Name.fillna('')
grp = pd.Series('一般', index=pk.index)
grp[name.str.contains('Mega') & (pk.Legendary == 0)] = 'Mega'
grp[name.isin(PSEUDO)] = '600族'
grp[pk.Legendary == 1] = '傳說'
print('\n== 傳說 / 600族 / 一般 ==')
print(pk.groupby(grp).agg(**agg).round(3).to_string())
