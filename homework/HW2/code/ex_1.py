"""
Εργασία 2 - Άσκηση 1
Ανάλυση ευαισθησίας από το βέλτιστο tableau του

    max z = 2x1 + x2 + 4x3 - x4
    (P1) x1 + 2x2 +  x3 -  3x4 <=  8
    (P2)     - x2 +  x3 +  2x4 <=  0
    (P3) 2x1 +7x2 - 5x3 - 10x4 <= 21 ,  x >= 0

Ο κώδικας:
  1. ανακατασκευάζει τον βασικό πίνακα B, τον B^-1 και όλο το tableau,
  2. για κάθε διαταραχή (α)-(δ) ελέγχει ΒΕΛΤΙΣΤΟΤΗΤΑ (γραμμή c_j - z_j)
     και ΕΦΙΚΤΟΤΗΤΑ (B^-1 b) και επιλέγει primal ή dual simplex,
  3. εκτελεί και ΤΥΠΩΝΕΙ όλες τις επαναλήψεις με ακριβή ρητή αριθμητική,
  4. επαληθεύει κάθε αποτέλεσμα με ανεξάρτητη επίλυση (scipy.linprog).
"""
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

nm = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
A0 = [[F(v) for v in r] for r in
      [[1, 2, 1, -3, 1, 0, 0], [0, -1, 1, 2, 0, 1, 0], [2, 7, -5, -10, 0, 0, 1]]]
m = 3
OPT_BASIS = [0, 3, 6]          # x1, x4, x7


def build(basis, c, b):
    """Πλήρες tableau + γραμμή (c_j - z_j) + τιμή z, με ακριβή αριθμητική."""
    T = [[A0[i][j] for j in range(7)] + [F(b[i])] for i in range(m)]
    for r, bi in enumerate(basis):
        p = next(k for k in range(r, m) if T[k][bi] != 0)
        T[r], T[p] = T[p], T[r]
        pv = T[r][bi]
        T[r] = [v / pv for v in T[r]]
        for k in range(m):
            if k != r and T[k][bi] != 0:
                f = T[k][bi]
                T[k] = [T[k][t] - f * T[r][t] for t in range(8)]
    cc = [F(v) for v in c] + [F(0)] * 3
    red = [cc[j] - sum(cc[basis[i]] * T[i][j] for i in range(m)) for j in range(7)]
    z = sum(cc[basis[i]] * T[i][7] for i in range(m))
    return T, red, z


def show(tag, basis, T, red, z):
    print(f"\n--- {tag}   Βάση = ({', '.join(nm[j] for j in basis)}),  z = {z}")
    print("       " + "".join(f"{v:>8}" for v in nm) + f"{'RHS':>8}")
    for i in range(m):
        print(f" {nm[basis[i]]:<6}" + "".join(f"{str(T[i][j]):>8}" for j in range(7))
              + f"{str(T[i][7]):>8}")
    print(" c-z   " + "".join(f"{str(red[j]):>8}" for j in range(7)) + f"{str(z):>8}")


def primal(basis, c, b, tag):
    it = 0
    while True:
        T, red, z = build(basis, c, b)
        show(f"{tag} - primal simplex, επανάληψη {it}", basis, T, red, z)
        cand = [j for j in range(7) if j not in basis and red[j] > 0]
        if not cand:
            print("   => όλα τα c_j - z_j <= 0 : ΒΕΛΤΙΣΤΟ")
            x = [F(0)] * 7
            for i, bi in enumerate(basis):
                x[bi] = T[i][7]
            print(f"   x* = ({', '.join(str(x[j]) for j in range(4))}),  z* = {z}")
            return basis, z
        e = max(cand, key=lambda j: red[j])
        rat = [(T[i][7] / T[i][e], i) for i in range(m) if T[i][e] > 0]
        mn = min(r for r, _ in rat)
        i0 = [i for r, i in rat if r == mn][0]
        print(f"   εισέρχεται {nm[e]} (c-z = {red[e]}); λόγοι: "
              + ", ".join(f"{nm[basis[i]]}: {T[i][7]}/{T[i][e]} = {T[i][7]/T[i][e]}"
                          for _, i in rat)
              + f"  ->  εξέρχεται {nm[basis[i0]]}"
              + ("   (ΕΚΦΥΛΙΣΜΕΝΟ pivot, λόγος 0)" if mn == 0 else ""))
        basis = basis[:i0] + [e] + basis[i0 + 1:]
        it += 1


def dual(basis, c, b, tag):
    it = 0
    while True:
        T, red, z = build(basis, c, b)
        show(f"{tag} - dual simplex, επανάληψη {it}", basis, T, red, z)
        neg = [i for i in range(m) if T[i][7] < 0]
        if not neg:
            print("   => όλα τα RHS >= 0 : ΕΦΙΚΤΟ και ΒΕΛΤΙΣΤΟ")
            x = [F(0)] * 7
            for i, bi in enumerate(basis):
                x[bi] = T[i][7]
            print(f"   x* = ({', '.join(str(x[j]) for j in range(4))}),  z* = {z}")
            return basis, z
        r = min(neg, key=lambda i: T[i][7])
        cand = [j for j in range(7) if j not in basis and T[r][j] < 0]
        if not cand:
            print(f"   => η γραμμή {nm[basis[r]]} έχει RHS < 0 και ΟΛΑ τα στοιχεία >= 0")
            print("      δεν υπάρχει επιτρεπτό pivot  ->  ΤΟ ΠΡΟΒΛΗΜΑ ΕΙΝΑΙ ΜΗ ΕΦΙΚΤΟ")
            return None, None
        e = min(cand, key=lambda j: abs(red[j] / T[r][j]))
        print(f"   εξέρχεται {nm[basis[r]]} (RHS = {T[r][7]}); λόγοι |(c-z)/a|: "
              + ", ".join(f"{nm[j]}: {abs(red[j]/T[r][j])}" for j in cand)
              + f"  ->  εισέρχεται {nm[e]}")
        basis = basis[:r] + [e] + basis[r + 1:]
        it += 1


def check(c, b, tag):
    r = linprog([-v for v in c], A_ub=[[float(x) for x in row[:4]] for row in A0],
                b_ub=[float(v) for v in b], bounds=[(0, None)] * 4)
    if r.status == 0:
        print(f"   [scipy] {tag}: x = {np.round(r.x, 6)},  z = {-r.fun:.6f}")
    else:
        print(f"   [scipy] {tag}: {r.message}")


# ---------------------------------------------------------- αρχικό tableau
c0, b0 = [2, 1, 4, -1], [8, 0, 21]
T, red, z = build(OPT_BASIS, c0, b0)
show("ΔΟΣΜΕΝΟ ΒΕΛΤΙΣΤΟ TABLEAU", OPT_BASIS, T, red, z)
Bmat = np.array([[float(A0[i][j]) for j in OPT_BASIS] for i in range(m)])
print("\n B  =\n", Bmat)
print(" B^-1 =\n", np.round(np.linalg.inv(Bmat), 6))
print(" B^-1 b =", np.round(np.linalg.inv(Bmat) @ np.array(b0, float), 6))
print(" σκιώδεις τιμές y = cB B^-1 =",
      np.round(np.array([2., -1., 0.]) @ np.linalg.inv(Bmat), 6))

for tag, c, b, kind in [("(α) c1 = 1", [1, 1, 4, -1], b0, "p"),
                        ("(β) c = [1,2,3,4]", [1, 2, 3, 4], b0, "p"),
                        ("(γ) b3 = 11", c0, [8, 0, 11], "d"),
                        ("(δ) b = [3,-2,4]", c0, [3, -2, 4], "d")]:
    print("\n" + "#" * 78)
    print("#  " + tag)
    print("#" * 78)
    (primal if kind == "p" else dual)(list(OPT_BASIS), c, b, tag)
    check(c, b, tag)
