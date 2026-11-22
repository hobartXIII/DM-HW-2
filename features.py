"""共用：讀資料庫、屬性相剋表、建立對戰特徵（Q1 決定，Q2～Q5 沿用）"""
import sqlite3
import numpy as np
import pandas as pd

STATS = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
TYPES = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 'Ground',
         'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy']

# 屬性相剋表（第六世代，資料庫最高到第六世代）：攻擊方 -> {防守方: 倍率}，沒寫的是 1 倍
_CHART = {
    'Normal': {'Rock': .5, 'Ghost': 0, 'Steel': .5},
    'Fire': {'Fire': .5, 'Water': .5, 'Grass': 2, 'Ice': 2, 'Bug': 2, 'Rock': .5, 'Dragon': .5, 'Steel': 2},
    'Water': {'Fire': 2, 'Water': .5, 'Grass': .5, 'Ground': 2, 'Rock': 2, 'Dragon': .5},
    'Electric': {'Water': 2, 'Electric': .5, 'Grass': .5, 'Ground': 0, 'Flying': 2, 'Dragon': .5},
    'Grass': {'Fire': .5, 'Water': 2, 'Grass': .5, 'Poison': .5, 'Ground': 2, 'Flying': .5, 'Bug': .5,
              'Rock': 2, 'Dragon': .5, 'Steel': .5},
    'Ice': {'Fire': .5, 'Water': .5, 'Grass': 2, 'Ice': .5, 'Ground': 2, 'Flying': 2, 'Dragon': 2, 'Steel': .5},
    'Fighting': {'Normal': 2, 'Ice': 2, 'Poison': .5, 'Flying': .5, 'Psychic': .5, 'Bug': .5, 'Rock': 2,
                 'Ghost': 0, 'Dark': 2, 'Steel': 2, 'Fairy': .5},
    'Poison': {'Grass': 2, 'Poison': .5, 'Ground': .5, 'Rock': .5, 'Ghost': .5, 'Steel': 0, 'Fairy': 2},
    'Ground': {'Fire': 2, 'Electric': 2, 'Grass': .5, 'Poison': 2, 'Flying': 0, 'Bug': .5, 'Rock': 2, 'Steel': 2},
    'Flying': {'Electric': .5, 'Grass': 2, 'Fighting': 2, 'Bug': 2, 'Rock': .5, 'Steel': .5},
    'Psychic': {'Fighting': 2, 'Poison': 2, 'Psychic': .5, 'Dark': 0, 'Steel': .5},
    'Bug': {'Fire': .5, 'Grass': 2, 'Fighting': .5, 'Poison': .5, 'Flying': .5, 'Psychic': 2, 'Ghost': .5,
            'Dark': 2, 'Steel': .5, 'Fairy': .5},
    'Rock': {'Fire': 2, 'Ice': 2, 'Fighting': .5, 'Ground': .5, 'Flying': 2, 'Bug': 2, 'Steel': .5},
    'Ghost': {'Normal': 0, 'Psychic': 2, 'Ghost': 2, 'Dark': .5},
    'Dragon': {'Dragon': 2, 'Steel': .5, 'Fairy': 0},
    'Dark': {'Fighting': .5, 'Psychic': 2, 'Ghost': 2, 'Dark': .5, 'Fairy': .5},
    'Steel': {'Fire': .5, 'Water': .5, 'Electric': .5, 'Ice': 2, 'Rock': 2, 'Steel': .5, 'Fairy': 2},
    'Fairy': {'Fire': .5, 'Fighting': 2, 'Poison': .5, 'Dragon': 2, 'Dark': 2, 'Steel': .5},
}


def effectiveness(att_types, def_types):
    """攻擊方用自己最有利的屬性招式打防守方的倍率（防守方雙屬性時倍率相乘）"""
    best = 0.0
    for a in att_types:
        m = 1.0
        for d in def_types:
            m *= _CHART[a].get(d, 1.0)
        best = max(best, m)
    return best


def load(path='pokemon.db'):
    con = sqlite3.connect(path)
    pk = pd.read_sql('select * from pokemon', con).set_index('#')
    cb = pd.read_sql('select * from combats', con)
    con.close()
    pk['Legendary'] = pk['Legendary'].astype(str).str.upper().eq('TRUE').astype(int)
    pk['Total'] = pk[STATS].sum(axis=1)
    pk['types'] = [tuple(x for x in (a, b) if isinstance(x, str)) for a, b in zip(pk['Type 1'], pk['Type 2'])]
    cb['y'] = (cb.Winner == cb.First_pokemon).astype(int)
    return pk, cb


# diff：Q1 分析用的候選欄位（數值差值、相剋），不作為模型特徵
FEATURES_DIFF = ['d_HP', 'd_Attack', 'd_Defense', 'd_Sp. Atk', 'd_Sp. Def', 'd_Speed',
            'first_faster', 'd_Legendary', 'eff_1on2', 'eff_2on1']
# prio：Q2～Q5 的模型特徵（順序就是 LIBSVM 的 index 1, 2, ...），依「屬性 → 速度 → 攻防（Atk / SpA 擇高對上對應的 Def / SpD）」的優先度排序，再補 HP、傳說
# 屬性：兩個方向的相剋倍率分開，另加「攻擊無效」旗標（0 倍與 1/4 倍取 log2 後都是 -2，靠旗標區分）
FEATURES_PRIO = ['eff_1on2', 'eff_2on1', 'immune_1on2', 'immune_2on1', 'first_faster', 'd_Speed',
                 'off_1on2', 'off_2on1', 'd_HP', 'd_Legendary']
SETS = {'diff': FEATURES_DIFF, 'prio': FEATURES_PRIO}
FEATURES = FEATURES_PRIO


def offense(att, dfn):
    """攻擊方 Atk、SpA 擇高，除以防守方對應的 Def 或 SpD，取 log2（1 = 打得動兩倍）"""
    phys = att['Attack'].to_numpy(float) >= att['Sp. Atk'].to_numpy(float)
    ratio = np.where(phys, att['Attack'].to_numpy(float) / dfn['Defense'].to_numpy(float),
                     att['Sp. Atk'].to_numpy(float) / dfn['Sp. Def'].to_numpy(float))
    return np.log2(ratio)


def build(pk, cb, which='diff'):
    """每場對戰一列：First - Second 的差值、相剋、攻防特徵；which 選 diff、prio 或 all（全部欄位）"""
    A, B = pk.loc[cb.First_pokemon], pk.loc[cb.Second_pokemon]
    X = pd.DataFrame(index=cb.index)
    for s in STATS:
        X[f'd_{s}'] = A[s].to_numpy() - B[s].to_numpy()
    X['first_faster'] = (X['d_Speed'] > 0).astype(int)       # 同速時後手必勝，所以同速記 0
    X['d_Legendary'] = A['Legendary'].to_numpy() - B['Legendary'].to_numpy()
    # 相剋倍率取 log2：0 倍（無效）記為 -2，0.25→-2、0.5→-1、1→0、2→1、4→2
    lg = lambda m: -2.0 if m == 0 else float(np.log2(m))
    X['eff_1on2'] = [lg(effectiveness(a, b)) for a, b in zip(A['types'], B['types'])]
    X['eff_2on1'] = [lg(effectiveness(b, a)) for a, b in zip(A['types'], B['types'])]
    X['immune_1on2'] = [int(effectiveness(a, b) == 0) for a, b in zip(A['types'], B['types'])]  # 先手打不到後手
    X['immune_2on1'] = [int(effectiveness(b, a) == 0) for a, b in zip(A['types'], B['types'])]  # 後手打不到先手
    X['eff_adv'] = X['eff_1on2'] - X['eff_2on1']             # >0 表示先手屬性占優（只用於分析）
    X['off_1on2'] = offense(A, B)
    X['off_2on1'] = offense(B, A)
    return X if which == 'all' else X[SETS[which]]
