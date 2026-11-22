"""Q3 5-fold cross-validation：把 Q2 的 training set（40,000）分成五個大小相近的子集（分層），
沿用 Q2 的五組參數，每組各跑一次完整的 5-fold CV（每組 5 個模型 M_X1～M_X5）。
混淆矩陣：五個 fold 加總；其他指標：五個 fold 的平均。
選出最佳參數後，用整個 training set 重新訓練，在 test set 上評估。"""
import json
import os
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import features as F
import svmtools as T

FS = 'prio'
S = np.load('split.npz')
TR, TE = S['train'], S['test']


def load_xy():
    pk, cb = F.load()
    return F.build(pk, cb, FS).to_numpy(float), cb.y.to_numpy()


def one_fold(job):
    """訓練一個 M_Xk：用其餘四個 fold 算縮放參數並訓練，在第 k 個 fold 上評估"""
    key, k, fit_idx, ev_idx = job
    X, y = load_xy()
    Xs = T.apply_scale(X, T.fit_scale(X[fit_idx]))
    par = T.param_str(key, y[fit_idx])
    t0 = time.time()
    _, pred, _ = T.train_predict(y[fit_idx].tolist(), Xs[fit_idx].tolist(),
                                 y[ev_idx].tolist(), Xs[ev_idx].tolist(), par)
    return {'set': key, 'fold': k + 1, 'param': par, 'sec': round(time.time() - t0, 1), **T.metrics(y[ev_idx], pred)}


if __name__ == '__main__':
    _, y = load_xy()
    folds = [TR[f] for f in T.stratified_folds(y[TR], k=5)]     # fold 內是 TR 的位置 → 轉成原始索引
    for i, f in enumerate(folds):
        print(f'fold {i + 1}: {len(f)} 筆，先手勝比例 {y[f].mean():.4f}')
    jobs = []
    for key in T.PARAMS:
        for k in range(5):
            fit = np.concatenate([folds[j] for j in range(5) if j != k])
            jobs.append((key, k, fit, folds[k]))
    order = sorted(range(len(jobs)), key=lambda i: {'D': 0, 'B': 1}.get(jobs[i][0], 2))  # 慢的先跑
    n = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    print(f'{len(jobs)} 個模型，{n} 個行程平行\n', flush=True)
    t0 = time.time()
    res = [None] * len(jobs)
    with Pool(n) as pool:
        for i, r in zip(order, pool.imap(one_fold, [jobs[i] for i in order])):
            res[i] = r
            print(f"{r['set']} fold {r['fold']}  acc {r['accuracy']:.4f}  ({r['sec']}s)", flush=True)
    print(f'總耗時 {time.time() - t0:.0f}s')

    per_fold = pd.DataFrame(res)
    per_fold.to_csv('q3_folds.csv', index=False, encoding='utf-8-sig')

    # 每組參數：混淆矩陣加總、指標取平均
    rows = []
    for key, g in per_fold.groupby('set', sort=False):
        row = {'set': key, 'param': g.param.iloc[0]}
        row.update({c: int(g[c].sum()) for c in ['TP', 'FN', 'FP', 'TN']})
        row.update({c: g[c].mean() for c in T.METRIC_COLS})
        row['acc_std'] = g.accuracy.std()
        rows.append(row)
    cv = pd.DataFrame(rows)
    cv.to_csv('q3_cv.csv', index=False, encoding='utf-8-sig')

    cv['gap'] = (cv.FPR - cv.FNR).abs()
    best = cv.sort_values(['accuracy', 'F1', 'gap'], ascending=[False, False, True]).iloc[0]
    json.dump({'features': FS, 'set': best.set, 'param': T.PARAMS[best.set]}, open('q3_best.json', 'w'))

    # 最佳參數：整個 training set 重訓，test 評估
    X, y = load_xy()
    Xs = T.apply_scale(X, T.fit_scale(X[TR]))
    par = T.param_str(best.set, y[TR])
    model, pred, _ = T.train_predict(y[TR].tolist(), Xs[TR].tolist(), y[TE].tolist(), Xs[TE].tolist(), par)
    fin = T.metrics(y[TE], pred)
    pd.DataFrame([{'set': best.set, 'param': par, 'nSV': model.get_nr_sv(), **fin}]) \
        .to_csv('q3_test.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    fmt = lambda d: d.assign(**{c: d[c].map(lambda v: f'{v:.4f}') for c in T.METRIC_COLS + ['acc_std'] if c in d})
    print('\n== 5-fold CV（混淆矩陣為五折加總，指標為五折平均）==')
    print(fmt(cv[['set', 'param', 'TP', 'FN', 'FP', 'TN'] + T.METRIC_COLS + ['acc_std']]).to_string(index=False))
    print('\n== 各 fold accuracy ==')
    print(per_fold.pivot_table(index='set', columns='fold', values='accuracy', sort=False).round(4).to_string())
    print(f'\n最佳：{best.set}（{best.param}）CV accuracy {best.accuracy:.4f} ± {best.acc_std:.4f}')
    print('\n== 最佳參數用整個 training set 重訓，test 評估 ==')
    print('  '.join(f'{k} {v}' if isinstance(v, int) else f'{k} {v:.4f}' for k, v in fin.items()))
