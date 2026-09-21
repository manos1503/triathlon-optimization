"""
Εργασία 1 - Άσκηση 5
    max Z = -2x1 + x2 - 4x3 + 3x4
    (P1) x1 + x2 + 3x3 + 2x4 <= 4
    (P2) x1      -  x3 +  x4 <= 2
    (P3) 2x1 + x2            <= 3
         x1..x4 >= 0

(α) Όλες οι τομές n=4 υπερεπιπέδων από τα m+n=7 -> C(7,4)=35 συνδυασμοί.
    Ξεχωρίζουμε ποιες είναι κορυφές του πολυτόπου και ποιες εκφυλισμένες.
(β) Με μεταβλητές χαλάρωσης s1,s2,s3 -> 7 μεταβλητές, 3 εξισώσεις.
    Όλες οι βασικές λύσεις = C(7,3) = 35 επιλογές βάσης.
(γ) Αντιστοίχιση βασικών λύσεων <-> κορυφών και βέλτιστη λύση.

Όλοι οι υπολογισμοί γίνονται με ΑΚΡΙΒΗ ρητή αριθμητική (fractions), ώστε να
μην υπάρχει καμία αμφιβολία για τον εντοπισμό των μηδενικών (εκφυλισμός).
"""
import itertools
from fractions import Fraction as F

A = [[1, 1, 3, 2], [1, 0, -1, 1], [2, 1, 0, 0]]
b = [4, 2, 3]
c = [-2, 1, -4, 3]
A = [[F(v) for v in r] for r in A]
b = [F(v) for v in b]
c = [F(v) for v in c]
m, n = 3, 4


def gauss(M, r):
    """Λύση τετραγωνικού συστήματος με ακριβή αριθμητική· None αν ιδιάζον."""
    k = len(r)
    M = [row[:] for row in M]
    r = r[:]
    for i in range(k):
        p = next((j for j in range(i, k) if M[j][i] != 0), None)
        if p is None:
            return None
        M[i], M[p] = M[p], M[i]
        r[i], r[p] = r[p], r[i]
        pv = M[i][i]
        M[i] = [v / pv for v in M[i]]
        r[i] /= pv
        for j in range(k):
            if j != i and M[j][i] != 0:
                f = M[j][i]
                M[j] = [M[j][t] - f * M[i][t] for t in range(k)]
                r[j] -= f * r[i]
    return r


# =====================================================================
# (α)  Τομές υπερεπιπέδων στον R^4
# =====================================================================
H = [(A[i][:], b[i], f"P{i+1}") for i in range(m)]
for j in range(n):
    e = [F(0)] * n
    e[j] = F(1)
    H.append((e, F(0), f"x{j+1}=0"))

print("=" * 78)
print("(α) ΤΟΜΕΣ 4 ΥΠΕΡΕΠΙΠΕΔΩΝ ΑΠΟ 7  ->  C(7,4) = 35")
print("=" * 78)
sing = []
pts = {}
inf = 0
for comb in itertools.combinations(range(7), 4):
    M = [H[i][0][:] for i in comb]
    rhs = [H[i][1] for i in comb]
    lab = " ∩ ".join(H[i][2] for i in comb)
    sol = gauss(M, rhs)
    if sol is None:
        sing.append(lab)
        continue
    feas = all(v >= 0 for v in sol) and \
        all(sum(A[i][j] * sol[j] for j in range(n)) <= b[i] for i in range(m))
    if feas:
        pts.setdefault(tuple(sol), []).append(lab)
    else:
        inf += 1
print(f"ιδιάζοντα (παράλληλα, χωρίς μοναδική τομή): {len(sing)}")
print(f"τομές που ΔΕΝ είναι εφικτές                : {inf}")
print(f"τομές που είναι ΕΦΙΚΤΕΣ                     : {sum(len(v) for v in pts.values())}")
print(f"ΔΙΑΚΡΙΤΕΣ κορυφές του πολυτόπου            : {len(pts)}\n")
for p in sorted(pts, key=lambda q: [float(v) for v in q]):
    act = [f"P{i+1}" for i in range(m)
           if sum(A[i][j] * p[j] for j in range(n)) == b[i]]
    act += [f"x{j+1}=0" for j in range(n) if p[j] == 0]
    z = sum(c[j] * p[j] for j in range(n))
    tag = "  <== ΕΚΦΥΛΙΣΜΕΝΗ" if len(act) > n else ""
    print(f"  x=({','.join(str(v) for v in p):<16}) Z={str(z):>6}  "
          f"ενεργά υπερεπίπεδα: {len(act):>2} {act}{tag}")

# =====================================================================
# (β)  Βασικές λύσεις του συστήματος με μεταβλητές χαλάρωσης
# =====================================================================
As = [A[i] + [F(1) if j == i else F(0) for j in range(m)] for i in range(m)]
cs = c + [F(0)] * m
nm = ["x1", "x2", "x3", "x4", "s1", "s2", "s3"]

print("\n" + "=" * 78)
print("(β) ΟΛΕΣ ΟΙ ΒΑΣΙΚΕΣ ΛΥΣΕΙΣ  ->  C(7,3) = 35 επιλογές βάσης")
print("=" * 78)
print(f"{'#':>3} {'Βάση':<14} " + "".join(f"{v:>7}" for v in nm) + f"{'Z':>8}  Χαρακτηρισμός")
bfs, deg, nb = 0, 0, 0
table = []
for k, comb in enumerate(itertools.combinations(range(7), 3), 1):
    M = [[As[i][j] for j in comb] for i in range(m)]
    sol = gauss(M, b)
    Bn = "{" + ",".join(nm[j] for j in comb) + "}"
    if sol is None:
        nb += 1
        print(f"{k:>3} {Bn:<14} " + " " * 49 + "  ιδιάζων B (δεν είναι βάση)")
        table.append((k, Bn, None, None, "ιδιάζων B"))
        continue
    x = [F(0)] * 7
    for t, j in enumerate(comb):
        x[j] = sol[t]
    feas = all(v >= 0 for v in x)
    d = feas and any(v == 0 for v in sol)
    z = sum(cs[j] * x[j] for j in range(7))
    st = ("ΒΕΦΛ" + (" ΕΚΦΥΛΙΣΜΕΝΗ" if d else "")) if feas else "μη-εφικτή βασική λύση"
    bfs += feas
    deg += d
    print(f"{k:>3} {Bn:<14} " + "".join(f"{str(v):>7}" for v in x) +
          f"{str(z):>8}  {st}")
    table.append((k, Bn, x, z, st))

print(f"\nΣΥΝΟΨΗ: 35 συνδυασμοί = {nb} ιδιάζοντες + {35-nb} βασικές λύσεις")
print(f"        από τις {35-nb} βασικές λύσεις οι {bfs} είναι ΕΦΙΚΤΕΣ (ΒΕΦΛ)")
print(f"        από τις {bfs} ΒΕΦΛ οι {deg} είναι ΕΚΦΥΛΙΣΜΕΝΕΣ")

# =====================================================================
# (γ)  Αντιστοίχιση και βέλτιστη λύση
# =====================================================================
print("\n" + "=" * 78)
print("(γ) ΑΝΤΙΣΤΟΙΧΙΣΗ ΒΕΦΛ  <->  ΚΟΡΥΦΩΝ")
print("=" * 78)
groups = {}
for k, Bn, x, z, st in table:
    if x is None or not all(v >= 0 for v in x):
        continue
    groups.setdefault(tuple(x[:4]), []).append(Bn)
for v, bs in sorted(groups.items(), key=lambda t: float(sum(c[j]*t[0][j] for j in range(4)))):
    z = sum(c[j] * v[j] for j in range(4))
    print(f"  κορυφή ({','.join(str(t) for t in v):<14}) Z={str(z):>6}  <- {len(bs)} βάση/βάσεις: {bs}")
best = max(groups, key=lambda v: sum(c[j] * v[j] for j in range(4)))
print(f"\nΒΕΛΤΙΣΤΗ ΚΟΡΥΦΗ: x* = {tuple(str(v) for v in best)}, "
      f"Z* = {sum(c[j]*best[j] for j in range(4))}")
print(f"Είναι ΕΚΦΥΛΙΣΜΕΝΗ: της αντιστοιχούν {len(groups[best])} διαφορετικές βάσεις.")

# επαλήθευση
from scipy.optimize import linprog
import numpy as np
r = linprog([2, -1, 4, -3], A_ub=[[1, 1, 3, 2], [1, 0, -1, 1], [2, 1, 0, 0]],
            b_ub=[4, 2, 3], bounds=[(0, None)] * 4)
print(f"\nΕπαλήθευση scipy.linprog: x={np.round(r.x,6)}, Z={-r.fun:.6f}")
