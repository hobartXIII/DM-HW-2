"""Q1 補充：完整屬性分布表與屬性相剋表（輸出文字結果，並產生報告用的 HTML 表格片段）"""
import numpy as np
import pandas as pd
import features as F

# 台灣官方譯名
ZH = {'Normal': '一般', 'Fire': '火', 'Water': '水', 'Grass': '草', 'Electric': '電', 'Ice': '冰',
      'Fighting': '格鬥', 'Poison': '毒', 'Ground': '地面', 'Flying': '飛行', 'Psychic': '超能力', 'Bug': '蟲',
      'Rock': '岩石', 'Ghost': '幽靈', 'Dragon': '龍', 'Dark': '惡', 'Steel': '鋼', 'Fairy': '妖精'}

pk, cb = F.load()
games = cb.First_pokemon.value_counts().add(cb.Second_pokemon.value_counts(), fill_value=0)
wins = cb.Winner.value_counts()
pk['games'] = games.reindex(pk.index).fillna(0)
pk['win_rate'] = (wins.reindex(pk.index).fillna(0) / pk.games).where(pk.games > 0)

# ---------- 1. 完整屬性分布 ----------
rows = []
for t in F.TYPES:
    has = pk['types'].map(lambda ts: t in ts)
    sub = pk[has]
    rows.append({'type': t, 'zh': ZH[t],
                 't1': int((pk['Type 1'] == t).sum()), 't2': int((pk['Type 2'] == t).sum()),
                 'single': int((has & pk['Type 2'].isna()).sum()),
                 'win': sub.win_rate.mean(), 'speed': sub.Speed.mean(), 'total': sub.Total.mean()})
dist = pd.DataFrame(rows)
dist['all'] = dist.t1 + dist.t2
dist = dist.sort_values('all', ascending=False).reset_index(drop=True)
print('== 完整屬性分布（含此屬性者：Type 1 或 Type 2）==')
print(dist.round(3).to_string(index=False))
print(f"單屬性 {pk['Type 2'].isna().sum()}、雙屬性 {pk['Type 2'].notna().sum()}、共 {len(pk)}")
r = np.corrcoef(dist.win, dist.speed)[0, 1]
print(f'各屬性平均勝率 與 平均 Speed 的相關 r = {r:.3f}')

# ---------- 2. 屬性相剋表 ----------
M = pd.DataFrame([[F._CHART[a].get(d, 1.0) for d in F.TYPES] for a in F.TYPES], index=F.TYPES, columns=F.TYPES)
print('\n== 屬性相剋表（列 = 攻擊方，欄 = 防守方）==')
print(M.rename(index=ZH, columns=ZH).to_string())
summ = pd.DataFrame({'zh': [ZH[t] for t in F.TYPES],
                     'atk_2x': (M == 2).sum(axis=1).to_numpy(),   # 打得動幾種屬性
                     'def_2x': (M == 2).sum(axis=0).to_numpy(),   # 被幾種屬性剋
                     'def_resist': (M < 1).sum(axis=0).to_numpy()}, index=F.TYPES)
print(summ.to_string())


# ---------- 3. 報告用 HTML 片段 ----------
def html_dist():
    h = ['<table>', '  <tr><th>屬性</th><th>英文</th><th>Type 1</th><th>Type 2</th><th>合計</th>'
                    '<th>其中單屬性</th><th>平均勝率</th><th>平均 Speed</th><th>平均 Total</th></tr>']
    for _, x in dist.iterrows():
        h.append(f"  <tr><td>{x.zh}</td><td>{x.type}</td><td>{x.t1}</td><td>{x.t2}</td><td>{x['all']}</td>"
                 f"<td>{x.single}</td><td>{x.win:.3f}</td><td>{x.speed:.1f}</td><td>{x.total:.0f}</td></tr>")
    h.append('</table>')
    return '\n'.join(h)


def html_chart():
    sym = {2.0: '2', 0.5: '½', 0.0: '0'}
    style = {2.0: ' class="se"', 0.5: ' class="nv"', 0.0: ' class="im"'}
    h = ['<table class="chart">',
         '  <tr><th>攻＼防</th>' + ''.join(f'<th>{ZH[t]}</th>' for t in F.TYPES) + '</tr>']
    for a in F.TYPES:
        cells = ''.join(f"<td{style.get(M.at[a, d], '')}>{sym.get(M.at[a, d], '')}</td>" for d in F.TYPES)
        h.append(f'  <tr><th>{ZH[a]}</th>{cells}</tr>')
    h.append('</table>')
    return '\n'.join(h)


with open('q1_types_tables.html', 'w', encoding='utf-8') as f:
    f.write('<!-- 完整屬性分布 -->\n' + html_dist() + '\n\n<!-- 屬性相剋表 -->\n' + html_chart() + '\n')
print('\n已輸出 q1_types_tables.html')
