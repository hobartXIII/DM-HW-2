"""Q4 ROC curve 與 AUC（自行實作，不使用 scikit-learn 等現成函式）
模型：Q3 選出的最佳參數（C：-t 2 -c 10 -g 10），加 -b 1 重新訓練以取得機率。
正類 = 先手勝（label 1），score = P(label = 1 | x)。"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_train, svm_predict
import features as F
import svmtools as T


# ---------- 自行實作的 ROC 與 AUC ----------
def roc_curve(y_true, score):
    """門檻從 +∞ 往下掃：每降到一個新的分數，分數 ≥ 門檻的都判為正類。
    同分的資料一起跨過門檻（只產生一個點），避免同分時順序影響曲線。
    回傳 FPR、TPR 陣列（含起點 (0,0) 與終點 (1,1)）與對應門檻。"""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    P, N = int((y_true == 1).sum()), int((y_true == 0).sum())
    order = np.argsort(-score, kind='mergesort')          # 分數由高到低
    s, y = score[order], y_true[order]
    fpr, tpr, thr = [0.0], [0.0], [np.inf]
    tp = fp = 0
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:                # 同分的一整群
            tp += y[j] == 1
            fp += y[j] == 0
            j += 1
        fpr.append(fp / N)
        tpr.append(tp / P)
        thr.append(s[i])
        i = j
    return np.array(fpr), np.array(tpr), np.array(thr)


def auc_trapezoid(fpr, tpr):
    """梯形法：相鄰兩點之間的面積 = 寬 (ΔFPR) × 平均高 ((TPR_i + TPR_{i+1}) / 2)"""
    area = 0.0
    for k in range(1, len(fpr)):
        area += (fpr[k] - fpr[k - 1]) * (tpr[k] + tpr[k - 1]) / 2
    return area


def auc_rank(y_true, score):
    """驗算：AUC = 隨機抽一個正類、一個負類，正類分數較高的機率（同分算 1/2）
    用排名（Mann–Whitney U）計算：AUC = (正類排名和 − P(P+1)/2) / (P·N)"""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):                                     # 同分給平均排名
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    P, N = int((y_true == 1).sum()), int((y_true == 0).sum())
    return (ranks[y_true == 1].sum() - P * (P + 1) / 2) / (P * N)


# ---------- 訓練與預測 ----------
if __name__ == '__main__':
    best = json.load(open('q3_best.json'))
    S = np.load('split.npz')
    tr, te = S['train'], S['test']
    pk, cb = F.load()
    y = cb.y.to_numpy()
    Xdf = F.build(pk, cb, best['features'])
    X = Xdf.to_numpy(float)
    Xs = T.apply_scale(X, T.fit_scale(X[tr]))
    T.write_libsvm(f"{T.OUT}/q4_train.scale", y[tr], Xs[tr])
    T.write_libsvm(f"{T.OUT}/q4_test.scale", y[te], Xs[te])
    ytr, xtr = T.read_libsvm(f"{T.OUT}/q4_train.scale")
    yte, xte = T.read_libsvm(f"{T.OUT}/q4_test.scale")

    par = T.param_str(best['set'], y[tr]) + ' -b 1'
    print('訓練參數:', par)
    model = svm_train(ytr, xtr, par + ' -q')
    p_lab, _, p_prob = svm_predict(yte, xte, model, '-b 1 -q')
    labels = model.get_labels()                           # 機率欄位的順序跟著 labels
    score = np.array(p_prob)[:, labels.index(1)]          # P(先手勝)
    yt = np.array(yte, dtype=int)

    fpr, tpr, thr = roc_curve(yt, score)
    auc = auc_trapezoid(fpr, tpr)
    auc2 = auc_rank(yt, score)
    print(f'labels 順序 {labels}；ROC 點數 {len(fpr)}（不同分數 {len(np.unique(score))} 個 + 起點）')
    print(f'AUC（梯形法）= {auc:.6f}   AUC（排名驗算）= {auc2:.6f}')

    # 門檻 0.5 的操作點、只看速度的規則點，用來在圖上對照
    pred05 = (score >= 0.5).astype(int)
    m05 = T.metrics(yt, pred05)
    mlab = T.metrics(yt, np.array(p_lab, dtype=int))
    speed = Xdf['first_faster'].to_numpy()[te].astype(int)
    msp = T.metrics(yt, speed)
    print(f"門檻 0.5：TPR {m05['recall(TPR)']:.4f} FPR {m05['FPR']:.4f} acc {m05['accuracy']:.4f}")
    print(f"svm-predict -b 1 的 label：acc {mlab['accuracy']:.4f}（TP {mlab['TP']} FN {mlab['FN']} FP {mlab['FP']} TN {mlab['TN']}）")
    print(f"只看速度規則：TPR {msp['recall(TPR)']:.4f} FPR {msp['FPR']:.4f}")
    # 左上角最近點（Youden's J = TPR − FPR 最大）
    k = int(np.argmax(tpr - fpr))
    print(f'Youden J 最大：門檻 {thr[k]:.4f}  TPR {tpr[k]:.4f}  FPR {fpr[k]:.4f}')
    np.savez('q4_scores.npz', y=yt, score=score, fpr=fpr, tpr=tpr, thr=thr)

    # ---------- 畫圖（matplotlib 只負責畫，座標點與面積都由上面的函式算出）----------
    plt.rcParams.update({'font.family': ['Microsoft JhengHei', 'sans-serif'], 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200,
                         'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
    BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, zoom in zip(axes, [False, True]):
        ax.plot([0, 1], [0, 1], ls='--', lw=1, color=MUTED, label='隨機猜測（AUC = 0.5）')
        ax.fill_between(fpr, tpr, step=None, color=BLUE, alpha=.10, lw=0)
        ax.plot(fpr, tpr, color=BLUE, lw=2, label=f'SVM（AUC = {auc:.4f}）')
        ax.scatter([m05['FPR']], [m05['recall(TPR)']], s=60, color=BLUE, edgecolor='white', lw=1.5, zorder=3,
                   label='門檻 0.5')
        ax.scatter([msp['FPR']], [msp['recall(TPR)']], s=60, marker='s', color=ORANGE, edgecolor='white', lw=1.5,
                   zorder=3, label='只看速度的規則')
        ax.set_xlabel('FPR（1 − specificity）')
        ax.set_ylabel('TPR（recall）')
        ax.grid(color='#e4e3df', lw=.8)
        ax.set_axisbelow(True)
        if zoom:
            ax.set_xlim(0, .3)
            ax.set_ylim(.7, 1.0)
            ax.set_title('放大左上角', fontsize=11, color=INK)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.01)
            ax.set_aspect('equal')
            ax.set_title(f"ROC curve（參數 {best['set']}，test set 10,000 筆）", fontsize=11, color=INK)
            ax.legend(loc='lower right', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig('fig_q4_roc.png')
    print('已輸出 fig_q4_roc.png')
