"""
Εργασία 2 - Άσκηση 4
Επίλυση με τον αλγόριθμο DUAL SIMPLEX του

    min z = 3x1 + 4x2 + 6x3 + 7x4 + x5
    (P1) 2x1 - x2 +  x3 + 6x4 - 5x5 >= 6
    (P2)  x1 + x2 + 2x3 +  x4 + 2x5 >= 3 ,   x >= 0

Πολλαπλασιάζουμε τους περιορισμούς με -1 και βάζουμε slack s1, s2:
η αρχική βάση {s1, s2} είναι ΔΥΪΚΑ εφικτή (όλα τα c_j >= 0 σε min) αλλά
ΠΡΩΤΕΥΟΝΤΩΣ μη εφικτή (RHS < 0) - ιδανική εκκίνηση για dual simplex.

Ο κώδικας τυπώνει κάθε επανάληψη σε μορφή tableau με ακριβή ρητή
αριθμητική, και στο τέλος ελέγχει εκφυλισμό, μοναδικότητα της βέλτιστης
λύσης και το σύνολο των βέλτιστων δυϊκών λύσεων.
"""
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

nm = ["x1", "x2", "x3", "x4", "x5", "s1", "s2"]
A = [[F(v) for v in r] for r in [[-2, 1, -1, -6, 5, 1, 0],
                                 [-1, -1, -2, -1, -2, 0, 1]]]
b = [F(-6), F(-3)]
c = [F(v) for v in [3, 4, 6, 7, 1, 0, 0]]
m = 2


def build(basis):
    T = [A[i][:] + [b[i]] for i in range(m)]
    for r, bi in enumerate(basis):
        p = next(k for k in range(r, m) if T[k][bi] != 0)
        T[r], T[p] = T[p], T[r]
        pv = T[r][bi]
        T[r] = [v / pv for v in T[r]]
        for k in range(m):
            if k != r and T[k][bi] != 0:
                f = T[k][bi]
                T[k] = [T[k][t] - f * T[r][t] for t in range(8)]
    red = [c[j] - sum(c[basis[i]] * T[i][j] for i in range(m)) for j in range(7)]
    z = sum(c[basis[i]] * T[i][7] for i in range(m))
    return T, red, z


def show(tag, basis, T, red, z):
    print(f"\n--- {tag}   Βάση = ({', '.join(nm[j] for j in basis)}),  z = {z}")
    print("       " + "".join(f"{v:>9}" for v in nm) + f"{'RHS':>9}")
    for i in range(m):
        print(f" {nm[basis[i]]:<6}" + "".join(f"{str(T[i][j]):>9}" for j in range(7))
              + f"{str(T[i][7]):>9}")
    print(" c-z   " + "".join(f"{str(red[j]):>9}" for j in range(7)) + f"{str(z):>9}")


basis = [5, 6]
it = 0
print("=" * 82)
print("DUAL SIMPLEX")
print("=" * 82)
while True:
    T, red, z = build(basis)
    show(f"Επανάληψη {it}", basis, T, red, z)
    neg = [i for i in range(m) if T[i][7] < 0]
    if not neg:
        print("\n  Όλα τα RHS >= 0 (πρωτεύουσα εφικτότητα) και όλα τα c_j - z_j >= 0")
        print("  (δυϊκή εφικτότητα)  =>  ΒΕΛΤΙΣΤΟ.")
        break
    r = min(neg, key=lambda i: T[i][7])
    print(f"  εξερχόμενη: {nm[basis[r]]} (πιο αρνητικό RHS = {T[r][7]})")
    cand = [j for j in range(7) if j not in basis and T[r][j] < 0]
    if not cand:
        print("  δεν υπάρχει αρνητικό στοιχείο στη γραμμή -> ΜΗ ΕΦΙΚΤΟ")
        break
    print("  ratio test |(c_j - z_j)/a_rj| μόνο για a_rj < 0: "
          + ", ".join(f"{nm[j]}: |{red[j]}/{T[r][j]}| = {abs(red[j]/T[r][j])}" for j in cand))
    e = min(cand, key=lambda j: abs(red[j] / T[r][j]))
    print(f"  ->  εισερχόμενη: {nm[e]},  στοιχείο οδηγός = {T[r][e]}")
    basis = basis[:r] + [e] + basis[r + 1:]
    it += 1

T, red, z = build(basis)
x = [F(0)] * 7
for i, bi in enumerate(basis):
    x[bi] = T[i][7]
print(f"\n  ΒΕΛΤΙΣΤΗ ΛΥΣΗ: x* = ({', '.join(str(x[j]) for j in range(5))}),  z* = {z}")
print(f"  P1: {2*x[0]-x[1]+x[2]+6*x[3]-5*x[4]} >= 6 ,  "
      f"P2: {x[0]+x[1]+2*x[2]+x[3]+2*x[4]} >= 3   (και οι δύο δεσμευτικοί)")

# ---------------------------------------------------- εκφυλισμός / μοναδικότητα
zb = [nm[basis[i]] for i in range(m) if T[i][7] == 0]
print(f"\n  ΕΚΦΥΛΙΣΜΟΣ: βασικές μεταβλητές με τιμή 0: {zb}")
print("    Στον R^5 μια κορυφή ορίζεται από 5 ενεργά υπερεπίπεδα· εδώ είναι ενεργά")
print("    τα x2=0, x3=0, x4=0, x5=0 ΚΑΙ οι δύο περιορισμοί P1, P2  ->  6 > 5,")
print("    άρα η βέλτιστη κορυφή είναι ΕΚΦΥΛΙΣΜΕΝΗ.")
nb = [j for j in range(7) if j not in basis]
print(f"  ΜΟΝΑΔΙΚΟΤΗΤΑ: c_j - z_j των μη βασικών = "
      f"{ {nm[j]: str(red[j]) for j in nb} }")
print("    όλα ΓΝΗΣΙΩΣ θετικά  =>  η βέλτιστη λύση είναι ΜΟΝΑΔΙΚΗ.")

# ------------------------------------------------------- δυϊκό & επαλήθευση
print("\n  ΔΥΪΚΟ:  max 6u1 + 3u2  s.t.  2u1+u2<=3, -u1+u2<=4, u1+2u2<=6,")
print("                                6u1+u2<=7, -5u1+2u2<=1,  u >= 0")
print("    Λόγω του εκφυλισμού της πρωτεύουσας κορυφής, το δυϊκό έχει ΠΟΛΛΑΠΛΕΣ")
print("    βέλτιστες λύσεις: για κάθε u1 στο [5/9, 1] με u2 = 3 - 2u1 ισχύει W = 9.")
for u1 in [F(5, 9), F(3, 4), F(1)]:
    u2 = 3 - 2 * u1
    ok = all(v <= 0 for v in [2*u1+u2-3, -u1+u2-4, u1+2*u2-6, 6*u1+u2-7, -5*u1+2*u2-1])
    print(f"      u = ({u1}, {u2}) εφικτή: {ok},  W = {6*u1+3*u2}")

r = linprog([3, 4, 6, 7, 1], A_ub=[[-2, 1, -1, -6, 5], [-1, -1, -2, -1, -2]],
            b_ub=[-6, -3], bounds=[(0, None)] * 5)
print(f"\n  [επαλήθευση scipy] x = {np.round(r.x,6)},  z = {r.fun:.6f}")
