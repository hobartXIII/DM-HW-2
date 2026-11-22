"""Q2～Q4 共用：分層切分、min-max 縮放（等同 svm-scale -l 0 -u 1）、LIBSVM 檔案、評估指標"""
import os
import numpy as np
from libsvm.svmutil import svm_read_problem, svm_train, svm_predict

SEED = 7761          # 課號，固定亂數種子讓結果可重現
OUT = 'libsvm_data'  # LIBSVM 格式檔案放這裡

# 五組參數組合（LIBSVM 參數字串）：線性、多項式、RBF、過度配適的 RBF、加類別權重的 RBF
PARAMS = {
    'A': '-t 0 -c 1',                  # 線性核
    'B': '-t 1 -d 3 -g 1 -c 10',       # 3 次多項式核
    'C': '-t 2 -c 10 -g 10',           # RBF，中等 C、gamma
    'D': '-t 2 -c 1000 -g 30',         # RBF，很大的 C、gamma（容易過度配適）
    'E': '-t 2 -c 10 -g 10 -w1 {w1}',  # 同 C，再用類別權重處理 class imbalance
}


def class_weight(y):
    """正類（label 1）比較少：w1 = 負類數 / 正類數，讓兩類的總懲罰相等"""
    y = np.asarray(y)
    return (y == 0).sum() / (y == 1).sum()


def param_str(key, y_train):
    return PARAMS[key].format(w1=f'{class_weight(y_train):.4f}')


def stratified_split(y, test_ratio=0.2, seed=SEED):
    """分層隨機切分：每個類別各自抽 20% 當 test，兩邊的類別比例和原資料相同"""
    rng = np.random.default_rng(seed)
    test = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        test.extend(idx[:round(len(idx) * test_ratio)])
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(y)), test)
    return train, test


def stratified_folds(y, k=5, seed=SEED):
    """把 y 的索引分成 k 個大小相近、類別比例相同的 fold"""
    rng = np.random.default_rng(seed + 1)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].extend(part)
    return [np.sort(np.array(f)) for f in folds]


def fit_scale(X, lo=0.0, hi=1.0):
    """只用訓練資料算每欄 min / max（svm-scale -s）"""
    mn, mx = X.min(axis=0), X.max(axis=0)
    return mn, mx, lo, hi


def apply_scale(X, p):
    """把參數套到任何資料（svm-scale -r）；常數欄位維持 lo"""
    mn, mx, lo, hi = p
    rng = np.where(mx > mn, mx - mn, 1.0)
    return lo + (hi - lo) * (X - mn) / rng


def write_libsvm(path, y, X):
    """寫成 <label> <index>:<value> ... 稀疏格式（0 不寫）"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='ascii', newline='\n') as f:
        for lab, row in zip(y, X):
            feats = ' '.join(f'{j + 1}:{v:.6g}' for j, v in enumerate(row) if v != 0)
            f.write(f'{int(lab)} {feats}\n')


def read_libsvm(path):
    return svm_read_problem(path)


def metrics(y_true, y_pred, pos=1):
    """混淆矩陣與七項指標；正類 = 先手勝（label 1）"""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    TP = int(((y_pred == pos) & (y_true == pos)).sum())
    FN = int(((y_pred != pos) & (y_true == pos)).sum())
    FP = int(((y_pred == pos) & (y_true != pos)).sum())
    TN = int(((y_pred != pos) & (y_true != pos)).sum())
    d = lambda a, b: a / b if b else 0.0
    m = {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
         'accuracy': d(TP + TN, TP + TN + FP + FN),
         'precision': d(TP, TP + FP),
         'recall(TPR)': d(TP, TP + FN),
         'FPR': d(FP, FP + TN),
         'TNR': d(TN, FP + TN),
         'FNR': d(FN, TP + FN)}
    m['F1'] = d(2 * m['precision'] * m['recall(TPR)'], m['precision'] + m['recall(TPR)'])
    return m


METRIC_COLS = ['accuracy', 'precision', 'recall(TPR)', 'FPR', 'TNR', 'FNR', 'F1']


def train_predict(ytr, xtr, yte, xte, param, extra=''):
    """用 LIBSVM 訓練並預測，回傳 (model, 預測 label, 預測值)"""
    model = svm_train(ytr, xtr, f'{param} {extra} -q')
    p_lab, _, p_val = svm_predict(yte, xte, model, '-q' + (' -b 1' if '-b 1' in extra else ''))
    return model, np.array(p_lab, dtype=int), p_val
