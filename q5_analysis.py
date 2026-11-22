"""Q5 預測結果分析：用 Q4 模型（參數 C，-b 1）在 test set 的預測（P ≥ 0.5 判先手勝）"""
import numpy as np
import pandas as pd
import features as F
import svmtools as T

pd.set_option('display.width', 220)
S = np.load('split.npz')
tr, te = S['train'], S['test']
Q = np.load('q4_scores.npz')
pk, cb = F.load()
X = F.build(pk, cb, 'all')   # 模型特徵 + 分析用欄位
y = cb.y.to_numpy()

D = X.iloc[te].copy().reset_index(drop=True)
D['y'] = y[te]
D['p'] = Q['score']
D['pred'] = (D.p >= 0.5).astype(int)
D['ok'] = D.pred == D.y
D['first'] = cb.First_pokemon.to_numpy()[te]
D['second'] = cb.Second_pokemon.to_numpy()[te]
D['speed_rel'] = np.select([D.d_Speed > 0, D.d_Speed < 0], ['先手較快', '先手較慢'], '同速')
D['speed_rule'] = D.first_faster
D['err_type'] = np.select([D.ok, D.pred == 1], ['正確', 'FP'], 'FN')
D['eff_adv'] = D.eff_1on2 - D.eff_2on1
D['adv'] = np.sign(D.eff_adv).map({-1: '屬性劣勢', 0: '屬性平手', 1: '屬性優勢'})
m = T.metrics(D.y, D.pred)
print(f"整體：acc {m['accuracy']:.4f}  TP {m['TP']} FN {m['FN']} FP {m['FP']} TN {m['TN']}  "
      f"precision {m['precision']:.4f} recall {m['recall(TPR)']:.4f}")

print('\n== 1. 依速度關係分組 ==')
g = D.groupby('speed_rel').agg(n=('y', 'size'), 先手勝率=('y', 'mean'), 預測先手勝=('pred', 'mean'),
                                acc=('ok', 'mean'), 錯誤數=('ok', lambda s: (~s).sum()))
print(g.round(4).to_string())

print('\n== 2. SVM vs 只看速度的規則：哪些對戰被改判 ==')
rule_ok = D.speed_rule == D.y
tab = pd.crosstab(rule_ok.map({True: '規則對', False: '規則錯'}), D.ok.map({True: 'SVM 對', False: 'SVM 錯'}))
print(tab.to_string())
fixed = D[~rule_ok & D.ok]
broke = D[rule_ok & ~D.ok]
print(f'SVM 救回 {len(fixed)} 筆、弄壞 {len(broke)} 筆，淨增 {len(fixed) - len(broke)} 筆')
print('救回的對戰：', fixed.speed_rel.value_counts().to_dict(), ' 屬性：', fixed.adv.value_counts().to_dict())

print('\n== 3. 錯誤集中在哪裡：先手較快時，依屬性優劣 ==')
fast = D[D.speed_rel == '先手較快']
print(fast.groupby('adv').agg(n=('y', 'size'), 先手實際勝率=('y', 'mean'), acc=('ok', 'mean'),
                              FP=('err_type', lambda s: (s == 'FP').sum())).round(4).to_string())
slow = D[D.speed_rel == '先手較慢']
print(slow.groupby('adv').agg(n=('y', 'size'), 先手實際勝率=('y', 'mean'), acc=('ok', 'mean'),
                              FN=('err_type', lambda s: (s == 'FN').sum())).round(4).to_string())

print('\n== 4. 先手較快：預測錯（FP）vs 預測對，特徵平均 ==')
cols = ['d_Speed', 'd_HP', 'd_Attack', 'd_Defense', 'd_Sp. Atk', 'd_Sp. Def', 'off_1on2', 'off_2on1', 'eff_1on2', 'eff_2on1', 'd_Legendary']
print(fast.groupby('err_type')[cols].mean().round(2).to_string())
print('\n先手較慢：預測錯（FN）vs 預測對')
print(slow.groupby('err_type')[cols].mean().round(2).to_string())

print('\n== 5. 速度差大小與錯誤率（較快的一方輸掉的比例）==')
D['abs_spd'] = D.d_Speed.abs()
bins = [0, 5, 20, 50, 200]
sub = D[D.speed_rel != '同速']
print(sub.groupby(pd.cut(sub.abs_spd, bins)).agg(n=('y', 'size'), acc=('ok', 'mean'),
      較快方輸=('speed_rule', lambda s: (s != D.loc[s.index, 'y']).mean())).round(4).to_string())

print('\n== 6. 模型信心與錯誤：依 P(先手勝) 分組 ==')
pb = pd.cut(D.p, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
print(D.groupby(pb).agg(n=('y', 'size'), 實際先手勝率=('y', 'mean'), acc=('ok', 'mean')).round(4).to_string())
print(f'錯誤的對戰中，P 落在 0.3～0.7 的比例：{D.loc[~D.ok, "p"].between(.3, .7).mean():.4f}；'
      f'P < 0.1 或 > 0.9 的高信心錯誤：{((~D.ok) & ((D.p < .1) | (D.p > .9))).sum()} 筆')

print('\n== 7. 資料本身的不一致：同一組對戰在資料庫中重複出現時，勝負是否一致 ==')
pair = cb.assign(a=cb.First_pokemon, b=cb.Second_pokemon)
pg = pair.groupby(['a', 'b']).y.agg(['size', 'mean'])
rep = pg[pg['size'] > 1]
print(f"重複的 (先手, 後手) 組合 {len(rep)} 組；其中勝負不一致的 {((rep['mean'] > 0) & (rep['mean'] < 1)).sum()} 組")
und = cb.assign(lo=np.minimum(cb.First_pokemon, cb.Second_pokemon), hi=np.maximum(cb.First_pokemon, cb.Second_pokemon))
ug = und.groupby(['lo', 'hi']).Winner.nunique()
print(f'不分先後手的同一對神奇寶貝：出現多次的 {(und.groupby(["lo", "hi"]).size() > 1).sum()} 對，'
      f'贏家不只一隻的 {(ug > 1).sum()} 對')
seen = set(zip(cb.First_pokemon.to_numpy()[tr], cb.Second_pokemon.to_numpy()[tr]))
D['seen'] = [(a, b) in seen for a, b in zip(D['first'], D['second'])]
print(D.groupby('seen').agg(n=('y', 'size'), acc=('ok', 'mean')).round(4).to_string())

print('\n== 8. 最常被判錯的神奇寶貝（test 中至少出場 15 次）==')
long = pd.concat([D[['first', 'ok']].rename(columns={'first': 'id'}), D[['second', 'ok']].rename(columns={'second': 'id'})])
e = long.groupby('id').ok.agg(['size', 'mean'])
e = e[e['size'] >= 15].sort_values('mean').head(8)
e['err_rate'] = 1 - e['mean']
e = e.join(pk[['Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']])
print(e.drop(columns='mean').round(3).to_string())

print('\n== 9. 傳說神奇寶貝對戰 ==')
D['leg'] = D.d_Legendary.map({1: '先手傳說', -1: '後手傳說', 0: '同級'})
print(D.groupby(['leg', 'speed_rel']).agg(n=('y', 'size'), acc=('ok', 'mean')).round(4).to_string())

print('\n== 10. 攻擊無效的對戰 ==')
D['immune'] = np.select([D.immune_1on2.eq(1) & D.immune_2on1.eq(1), D.immune_1on2.eq(1), D.immune_2on1.eq(1)],
                        ['雙方都打不到', '先手打不到', '後手打不到'], '都打得到')
print(D.groupby(['immune', 'speed_rel']).agg(n=('y', 'size'), 先手實際勝率=('y', 'mean'), 預測先手勝=('pred', 'mean'),
      acc=('ok', 'mean'), rule_acc=('speed_rule', lambda s: (s == D.loc[s.index, 'y']).mean())).round(4).to_string())

D.to_csv('q5_test_predictions.csv', index=False, encoding='utf-8-sig')

print('\n== 11. 鋼系神奇寶貝 ==')
games = cb.First_pokemon.value_counts().add(cb.Second_pokemon.value_counts(), fill_value=0)
pk['wr'] = cb.Winner.value_counts().reindex(pk.index).fillna(0) / games.reindex(pk.index)
pk['steel'] = pk['types'].map(lambda t: 'Steel' in t)
print(pk.groupby('steel').agg(n=('wr', 'size'), win_rate=('wr', 'mean'), speed=('Speed', 'mean'),
                              defense=('Defense', 'mean'), total=('Total', 'mean')).round(3).to_string())
pk['speed_bin'] = pd.cut(pk.Speed, [0, 50, 70, 90, 200], labels=['≤50', '51~70', '71~90', '>90'])
print(pk.groupby(['speed_bin', 'steel'], observed=True).wr.agg(['size', 'mean']).round(3).unstack().to_string())
# 對戰層級：每場拆成兩方，排除鋼系對鋼系
A, B = pk.loc[cb.First_pokemon].reset_index(drop=True), pk.loc[cb.Second_pokemon].reset_index(drop=True)
side = pd.concat([pd.DataFrame({'steel': me.steel.values, 'op_steel': op.steel.values, 'win': win,
                                'rel': np.select([me.Speed.values > op.Speed.values, me.Speed.values < op.Speed.values],
                                                 ['較快', '較慢'], '同速')})
                  for me, op, win in [(A, B, cb.y.values == 1), (B, A, cb.y.values == 0)]])
side = side[~(side.steel & side.op_steel)]
print(side.groupby(['rel', 'steel']).win.agg(['size', 'mean']).round(3).unstack().to_string())
has = pk.steel.reindex(D['first']).to_numpy() | pk.steel.reindex(D['second']).to_numpy()
print(f'test 中有鋼系的對戰 {has.sum()} 場 acc {D.ok[has].mean():.4f}；沒有鋼系 {(~has).sum()} 場 acc {D.ok[~has].mean():.4f}')

print('\n== 12. 皮卡丘遇到打不到的對手（電系打地面系無效）==')
Xall = F.build(pk, cb, 'all')
pid = int(pk.index[pk.Name == 'Pikachu'][0])
k1, k2 = cb.First_pokemon == pid, cb.Second_pokemon == pid
imm = (k1 & Xall.immune_1on2.eq(1)) | (k2 & Xall.immune_2on1.eq(1))
opp = np.where(cb.First_pokemon == pid, cb.Second_pokemon, cb.First_pokemon)[imm]
print(pd.DataFrame({'對手': pk.loc[opp, 'Name'].values, '對手 Speed': pk.loc[opp, 'Speed'].values,
                    '皮卡丘 Speed': pk.at[pid, 'Speed'], '皮卡丘是先手': k1[imm].values,
                    '皮卡丘獲勝': (cb.Winner[imm] == pid).values}).to_string(index=False))
