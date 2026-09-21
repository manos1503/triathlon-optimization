"""
Εργασία 1 - Άσκηση 7
Τεχνική Φάσης Ι του αλγορίθμου Simplex δύο φάσεων.

(α)  -5x1 +  x2 - 3x3 + 3x4 + 8x5 <= -3
      3x1 -  x2 + 2x3 -  x4 - 5x5 <=  2
     -2x1 +  x2 -  x3        + 3x5 <=  2 ,  x >= 0

(β)   x1 +  x2 +  x3 +  x4 +  x5 =  2
     -x1 + 2x2 +  x3 - 3x4 +  x5 =  1
      x1 - 3x2 - 2x3 + 2x4 - 2x5 = -4 ,  x >= 0

Στόχος Φάσης Ι:  min W' = Σ A_i  (ισοδύναμα max W = -Σ A_i).
Το σύστημα είναι εφικτό  <=>  W* = 0 (όλες οι τεχνητές μηδέν).
Ο κώδικας τυπώνει όλα τα tableaux με ακριβή ρητή αριθμητική και, σε
περίπτωση μη εφικτότητας, εξάγει και το πιστοποιητικό Farkas από τη
γραμμή W (γραμμικός συνδυασμός των περιορισμών που δίνει αντίφαση).
"""
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog


def phase1(rows, rhs, senses, title):
    """rows/rhs/senses: το αρχικό σύστημα. Επιστρέφει (εφικτό;, BFS)."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    n = len(rows[0])
    # 1) κάνουμε όλα τα b >= 0
    R, B, S = [], [], []
    for r, bb, s in zip(rows, rhs, senses):
        if bb < 0:
            R.append([-F(v) for v in r]); B.append(-F(bb))
            S.append({"<=": ">=", ">=": "<=", "=": "="}[s])
        else:
            R.append([F(v) for v in r]); B.append(F(bb)); S.append(s)
        print(f"  {'+'.join(f'{v}x{j+1}' for j,v in enumerate(R[-1]))} {S[-1]} {B[-1]}")
    # 2) slack / surplus / artificial
    cols, names, basis = [], [f"x{j+1}" for j in range(n)], []
    extra = []
    for i, s in enumerate(S):
        if s == "<=":                       # slack -> μπαίνει στη βάση
            extra.append(("s", i, F(1)));
        elif s == ">=":                     # surplus (-1) + τεχνητή
            extra.append(("e", i, F(-1)))
    # χτίσιμο πίνακα
    T = [R[i][:] for i in range(len(R))]
    for kind, i, sgn in extra:
        names.append(f"{'s' if kind=='s' else 'e'}{i+1}")
        for k in range(len(T)):
            T[k].append(sgn if k == i else F(0))
    art = []
    for i, s in enumerate(S):
        if s in (">=", "="):
            names.append(f"A{i+1}"); art.append(len(names) - 1)
            for k in range(len(T)):
                T[k].append(F(1) if k == i else F(0))
    # αρχική βάση
    for i, s in enumerate(S):
        if s == "<=":
            basis.append(names.index(f"s{i+1}"))
        else:
            basis.append(names.index(f"A{i+1}"))
    m, N = len(T), len(names)
    tab = [T[i] + [B[i]] for i in range(m)]

    def wrow():
        # W = -Σ A_i  -> γραμμή (c_j - z_j) με c_j = -1 για τεχνητές, 0 αλλού
        cw = [F(-1) if j in art else F(0) for j in range(N)]
        red = [cw[j] - sum(cw[basis[i]] * tab[i][j] for i in range(m)) for j in range(N)]
        w = sum(cw[basis[i]] * tab[i][N] for i in range(m))
        return red, w

    it = 0
    while True:
        red, w = wrow()
        print(f"\n--- Φάση Ι, επανάληψη {it}   Βάση = ({', '.join(names[j] for j in basis)}),  W = {w}")
        print("       " + "".join(f"{v:>7}" for v in names) + f"{'RHS':>7}")
        for i in range(m):
            print(f" {names[basis[i]]:<6}" + "".join(f"{str(tab[i][j]):>7}" for j in range(N))
                  + f"{str(tab[i][N]):>7}")
        print(" c-z   " + "".join(f"{str(red[j]):>7}" for j in range(N)) + f"{str(w):>7}")
        cand = [j for j in range(N) if j not in basis and red[j] > 0]
        if not cand:
            break
        e = max(cand, key=lambda j: red[j])
        rat = [(tab[i][N] / tab[i][e], i) for i in range(m) if tab[i][e] > 0]
        if not rat:
            print("  μη φραγμένο (δεν συμβαίνει στη Φάση Ι)"); return None
        mn = min(r for r, _ in rat)
        i0 = [i for r, i in rat if r == mn][0]
        print(f"  εισερχόμενη {names[e]} (c-z={red[e]}); λόγοι: "
              + ", ".join(f"{names[basis[i]]}: {tab[i][N]}/{tab[i][e]} = {tab[i][N]/tab[i][e]}"
                          for _, i in rat)
              + f"  ->  εξερχόμενη {names[basis[i0]]}")
        pv = tab[i0][e]
        tab[i0] = [v / pv for v in tab[i0]]
        for k in range(m):
            if k != i0 and tab[k][e] != 0:
                f = tab[k][e]
                tab[k] = [tab[k][t] - f * tab[i0][t] for t in range(N + 1)]
        basis[i0] = e
        it += 1

    red, w = wrow()
    if w == 0:
        x = [F(0)] * N
        for i, bi in enumerate(basis):
            x[bi] = tab[i][N]
        print(f"\n  W* = 0  =>  ΤΟ ΣΥΣΤΗΜΑ ΕΙΝΑΙ ΕΦΙΚΤΟ.")
        print("  Αρχική ΒΕΦΛ για τη Φάση ΙΙ: "
              + ", ".join(f"{names[j]}={x[j]}" for j in range(N) if x[j] != 0)
              + "  (όλες οι υπόλοιπες = 0)")
        print(f"  δηλαδή x = ({', '.join(str(x[j]) for j in range(n))})")
        return True
    print(f"\n  W* = {w} < 0  =>  ΤΟ ΣΥΣΤΗΜΑ ΕΙΝΑΙ ΜΗ ΕΦΙΚΤΟ.")
    inb = [names[j] for j in basis if j in art]
    print(f"  τεχνητές που παραμένουν στη βάση με θετική τιμή: {inb}")
    print("  (η γραμμή c-z των τεχνητών δίνει τους πολλαπλασιαστές του "
          "πιστοποιητικού μη-εφικτότητας κατά Farkas)")
    return False


# ---------------------------------------------------------------- (α)
phase1([[-5, 1, -3, 3, 8], [3, -1, 2, -1, -5], [-2, 1, -1, 0, 3]],
       [-3, 2, 2], ["<=", "<=", "<="],
       "ΣΥΣΤΗΜΑ (α)")
r = linprog([0]*5, A_ub=[[-5, 1, -3, 3, 8], [3, -1, 2, -1, -5], [-2, 1, -1, 0, 3]],
            b_ub=[-3, 2, 2], bounds=[(0, None)]*5)
print(f"  Επαλήθευση scipy: status={r.status} ({'εφικτό' if r.status==0 else 'μη εφικτό'})")

# ---------------------------------------------------------------- (β)
phase1([[1, 1, 1, 1, 1], [-1, 2, 1, -3, 1], [1, -3, -2, 2, -2]],
       [2, 1, -4], ["=", "=", "="],
       "ΣΥΣΤΗΜΑ (β)")
r = linprog([0]*5, A_eq=[[1, 1, 1, 1, 1], [-1, 2, 1, -3, 1], [1, -3, -2, 2, -2]],
            b_eq=[2, 1, -4], bounds=[(0, None)]*5)
print(f"  Επαλήθευση scipy: status={r.status} ({'εφικτό' if r.status==0 else 'μη εφικτό'})")

# Αλγεβρικό πιστοποιητικό μη-εφικτότητας για το (β) (κατά Farkas).
# Οι συντελεστές των τεχνητών στην τελική γραμμή του A3 δίνουν τους
# πολλαπλασιαστές· εδώ προκύπτει ο απλούστατος συνδυασμός E1 + E2 + E3.
E = np.array([[1, 1, 1, 1, 1], [-1, 2, 1, -3, 1], [1, -3, -2, 2, -2]])
rhs = np.array([2, 1, -4])
mult = np.array([1, 1, 1])
print("\n  ΠΙΣΤΟΠΟΙΗΤΙΚΟ ΜΗ-ΕΦΙΚΤΟΤΗΤΑΣ (Farkas) για το σύστημα (β):")
print(f"    E1 + E2 + E3  ->  συντελεστές {mult @ E}  =  {mult @ rhs}")
print("    δηλαδή  x1 = -1 < 0,  που αντιβαίνει στον περιορισμό x1 >= 0.")
print("    Άρα κανένα x >= 0 δεν ικανοποιεί ταυτόχρονα τις τρεις εξισώσεις.")
