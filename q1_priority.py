import numpy as np, features as F
pk, cb = F.load(); X = F.build(pk, cb); y = cb.y.to_numpy()
eff_adv = (X.eff_1on2 - X.eff_2on1).to_numpy()
dspe = X.d_Speed.to_numpy()
acc = lambda p: round(float((p == y).mean()), 4)
print('R1 速度:', acc(dspe > 0))
print('R2 屬性→速度:', acc(np.where(eff_adv != 0, eff_adv > 0, dspe > 0)))
print('R3 速度→屬性:', acc(np.where(dspe != 0, dspe > 0, eff_adv > 0)))
def fisher(x):
    mu = x.mean(); n = sum((y==c).sum()*(x[y==c].mean()-mu)**2 for c in (0,1)); d = sum((y==c).sum()*x[y==c].var() for c in (0,1)); return n/d
for n, v in {'eff_adv': eff_adv}.items(): print('Fisher', n, round(fisher(v), 4))
P = F.build(pk, cb, 'prio')
off1, off2 = P.off_1on2.to_numpy(), P.off_2on1.to_numpy()
print('R4 屬性→速度→攻防:', acc(np.where(eff_adv != 0, eff_adv > 0, np.where(dspe != 0, dspe > 0, off1 > off2))))
print('R5 只看攻防比(擇高):', acc(off1 > off2))
print('Fisher off_1on2 - off_2on1', round(fisher(off1 - off2), 4))
