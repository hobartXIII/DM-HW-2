"""Q2 Hold-out：80% training set / 20% test set（分層隨機切分）
特徵：Q1 決定的 10 維特徵（屬性 → 速度 → 攻防）。
五組參數各用整個 training set 訓練、在 test set 上評估，比較混淆矩陣與各項指標後選出最佳參數。"""
import json
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import features as F
import svmtools as T

FS = 'prio'


def load():
    pk, cb = F.load()
    return F.build(pk, cb, FS).to_numpy(float), cb.y.to_numpy()


def one(key):
    """用 training set 訓練一組參數，在 test set 上評估"""
    ytr, xtr = T.read_libsvm(f'{T.OUT}/q2_train.scale')
    yte, xte = T.read_libsvm(f'{T.OUT}/q2_test.scale')
    par = T.param_str(key, ytr)
    t0 = time.time()
    model, pred, _ = T.train_predict(ytr, xtr, yte, xte, par)
    return {'set': key, 'param': par, 'nSV': model.get_nr_sv(), 'sec': round(time.time() - t0, 1),
            **T.metrics(yte, pred)}


if __name__ == '__main__':
    X, y = load()
    tr, te = T.stratified_split(y)                         # 80 / 20
    np.savez('split.npz', train=tr, test=te)               # Q3、Q4、Q5 沿用同一個切分
    for n, idx in [('全部', np.arange(len(y))), ('training set', tr), ('test set', te)]:
        print(f'{n:12} {len(idx):6} 筆  先手勝 {(y[idx] == 1).sum():6}  後手勝 {(y[idx] == 0).sum():6}  '
              f'先手勝比例 {y[idx].mean():.4f}')
    print(f'class weight w1 = 負類數 / 正類數 = {T.class_weight(y[tr]):.4f}\n')

    # min-max 縮放只用 training set 的 min / max（等同 svm-scale -s，再用 -r 套到 test）
    Xs = T.apply_scale(X, T.fit_scale(X[tr]))
    T.write_libsvm(f'{T.OUT}/q2_train.txt', y[tr], X[tr])
    T.write_libsvm(f'{T.OUT}/q2_test.txt', y[te], X[te])
    T.write_libsvm(f'{T.OUT}/q2_train.scale', y[tr], Xs[tr])
    T.write_libsvm(f'{T.OUT}/q2_test.scale', y[te], Xs[te])

    keys = list(T.PARAMS)
    with Pool(len(keys)) as pool:
        rows = pool.map(one, keys)
    df = pd.DataFrame(rows)
    df.to_csv('q2_results.csv', index=False, encoding='utf-8-sig')

    # 選擇：accuracy 最高；相同時比 F1；再相同時選 FPR 與 FNR 差距較小者
    df['gap'] = (df.FPR - df.FNR).abs()
    best = df.sort_values(['accuracy', 'F1', 'gap'], ascending=[False, False, True]).iloc[0]
    json.dump({'features': FS, 'set': best.set, 'param': T.PARAMS[best.set]}, open('q2_best.json', 'w'))

    pd.set_option('display.width', 250)
    show = df[['set', 'param', 'TP', 'FN', 'FP', 'TN'] + T.METRIC_COLS + ['nSV', 'sec']].copy()
    for c in T.METRIC_COLS:
        show[c] = show[c].map(lambda v: f'{v:.4f}')
    print('== 五組參數：training set（40,000）訓練，test set（10,000）評估 ==')
    print(show.to_string(index=False))
    print(f'\n最佳：{best.set}（{best.param}）accuracy {best.accuracy:.4f}  F1 {best.F1:.4f}')
