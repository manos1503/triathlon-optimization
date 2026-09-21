"""
Εργασία 1 - Άσκηση 6
    max Z = 3x1 + 5x2 + 6x3
    2x1 + x2 + x3 <= 4 ;  x1 + 2x2 + x3 <= 4
    x1 + x2 + 2x3 <= 4 ;  x1 +  x2 +  x3 <= 3 ;  x >= 0

(α) Simplex με τυπωμένα tableaux σε ακριβή ρητή αριθμητική.
(β) ΠΛΗΡΗΣ διερεύνηση όλων των εναλλακτικών διαδρομών: σε κάθε κόμβο
    δοκιμάζονται ΟΛΕΣ οι εισερχόμενες με θετικό (c_j - z_j) και ΟΛΕΣ οι
    ισοπαλίες του ratio test. Παράγεται ο Simplex adjacency graph (κόμβοι =
    βάσεις/κορυφές, ακμές = επιτρεπτά pivots) και εντοπίζονται οι
    εκφυλισμένες κορυφές (περισσότερες από μία βάσεις στο ίδιο σημείο) και
    τα εκφυλισμένα pivots (αλλαγή βάσης χωρίς μεταβολή του σημείου / του Z).
"""
from fractions import Fraction as F

A = [[2, 1, 1], [1, 2, 1], [1, 1, 2], [1, 1, 1]]
b = [4, 4, 4, 3]
c = [3, 5, 6]
m, n = 4, 3
A = [[F(v) for v in r] for r in A]
b = [F(v) for v in b]
cost = [F(v) for v in c] + [F(0)] * m
M = [A[i] + [F(1) if j == i else F(0) for j in range(m)] for i in range(m)]
nm = ["x1", "x2", "x3", "s1", "s2", "s3", "s4"]


def build(basis):
    """Πλήρες tableau για τη δοσμένη βάση (Gauss-Jordan, ακριβής αριθμητική)."""
    T = [[M[i][j] for j in range(7)] + [b[i]] for i in range(m)]
    for r, bi in enumerate(basis):
        p = next((k for k in range(r, m) if T[k][bi] != 0), None)
        if p is None:
            return None
        T[r], T[p] = T[p], T[r]
        pv = T[r][bi]
        T[r] = [v / pv for v in T[r]]
        for k in range(m):
            if k != r and T[k][bi] != 0:
                f = T[k][bi]
                T[k] = [T[k][t] - f * T[r][t] for t in range(8)]
    # γραμμή κριτηρίου: c_j - z_j  (max: βέλτιστο όταν όλα <= 0)
    red = [cost[j] - sum(cost[basis[i]] * T[i][j] for i in range(m)) for j in range(7)]
    z = sum(cost[basis[i]] * T[i][7] for i in range(m))
    return T, red, z


def point(basis, T):
    x = [F(0)] * 7
    for i, bi in enumerate(basis):
        x[bi] = T[i][7]
    return tuple(x[:3])


def show(tag, basis, T, red, z):
    print(f"\n--- {tag}   Βάση = ({', '.join(nm[j] for j in basis)}),  Z = {z}")
    print("       " + "".join(f"{v:>8}" for v in nm) + f"{'RHS':>8}")
    for i in range(m):
        print(f" {nm[basis[i]]:<6}" + "".join(f"{str(T[i][j]):>8}" for j in range(7))
              + f"{str(T[i][7]):>8}")
    print(" c-z   " + "".join(f"{str(red[j]):>8}" for j in range(7)) + f"{str(z):>8}")


# =====================================================================
print("=" * 78)
print("(α) ΕΚΤΕΛΕΣΗ SIMPLEX (κανόνας Dantzig: μεγαλύτερο c_j - z_j)")
print("=" * 78)
basis = [3, 4, 5, 6]
it = 0
while True:
    T, red, z = build(basis)
    show(f"Επανάληψη {it}", basis, T, red, z)
    cand = [j for j in range(7) if j not in basis and red[j] > 0]
    if not cand:
        print("\n  Όλα τα c_j - z_j <= 0  =>  ΒΕΛΤΙΣΤΟ")
        print(f"  x* = {tuple(str(v) for v in point(basis, T))},  Z* = {z}")
        break
    e = max(cand, key=lambda j: red[j])
    rat = [(T[i][7] / T[i][e], i) for i in range(m) if T[i][e] > 0]
    mn = min(r for r, _ in rat)
    ties = [i for r, i in rat if r == mn]
    print(f"  εισερχόμενη: {nm[e]} (c-z = {red[e]})")
    print("  ratio test: " + ", ".join(f"{nm[basis[i]]}: {T[i][7]}/{T[i][e]} = {T[i][7]/T[i][e]}"
                                       for _, i in rat)
          + f"   ->  εξερχόμενη: {nm[basis[ties[0]]]}"
          + ("   (ΙΣΟΠΑΛΙΑ -> εκφυλισμός στην επόμενη κορυφή)" if len(ties) > 1 else ""))
    basis = basis[:ties[0]] + [e] + basis[ties[0] + 1:]
    it += 1

# =====================================================================
print("\n" + "=" * 78)
print("(β) SIMPLEX ADJACENCY GRAPH - ΟΛΕΣ ΟΙ ΕΝΑΛΛΑΚΤΙΚΕΣ ΔΙΑΔΡΟΜΕΣ")
print("=" * 78)
start = (3, 4, 5, 6)
seen, stack, edges, info = set(), [start], [], {}
while stack:
    bs = stack.pop()
    if bs in seen:
        continue
    seen.add(bs)
    T, red, z = build(list(bs))
    info[bs] = (point(list(bs), T), z)
    for j in range(7):
        if j in bs or red[j] <= 0:
            continue
        rat = [(T[i][7] / T[i][j], i) for i in range(m) if T[i][j] > 0]
        if not rat:
            continue
        mn = min(r for r, _ in rat)
        for r, i in rat:
            if r != mn:
                continue
            nb = tuple(sorted([x for x in bs if x != bs[i]] + [j]))
            edges.append((bs, nb, nm[j], nm[bs[i]], r == 0))
            if nb not in seen:
                stack.append(nb)

# ονομασία κορυφών
label, k = {}, 0
for bs in sorted(info, key=lambda t: float(info[t][1])):
    p = info[bs][0]
    if p not in label:
        label[p] = "OABCDEFGHIJ"[k]
        k += 1
print("\nΚΟΡΥΦΕΣ (κόμβοι του γράφου):")
for p in sorted(label, key=lambda q: float(sum(c[i] * q[i] for i in range(3)))):
    bl = [bs for bs in info if info[bs][0] == p]
    z = sum(F(c[i]) * p[i] for i in range(3))
    tag = f"   <== ΕΚΦΥΛΙΣΜΕΝΗ ({len(bl)} βάσεις)" if len(bl) > 1 else ""
    print(f"  {label[p]}: ({', '.join(str(v) for v in p)})  Z = {z}   βάσεις: "
          + ", ".join("{" + ",".join(nm[j] for j in bs) + "}" for bs in bl) + tag)

print("\nΑΚΜΕΣ (επιτρεπτά pivots):")
for a, bb, ein, lout, degen in sorted(edges, key=lambda e: (float(info[e[0]][1]), e[2])):
    pa, za = info[a]
    pb, zb = info[bb]
    tag = "  [ΕΚΦΥΛΙΣΜΕΝΟ pivot: ίδιο σημείο, ίδιο Z]" if pa == pb else ""
    print(f"  {label[pa]}{{{','.join(nm[j] for j in a)}}} --({ein} in / {lout} out)--> "
          f"{label[pb]}{{{','.join(nm[j] for j in bb)}}}   Z: {za} -> {zb}{tag}")

print(f"\nΣυνολικά: {len(label)} κορυφές, {len(info)} βάσεις, {len(edges)} επιτρεπτά pivots.")
