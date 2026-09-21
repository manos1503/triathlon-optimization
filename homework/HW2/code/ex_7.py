"""
Εργασία 2 - Άσκηση 7  (πρόβλημα σακιδίου 0-1)

    max z = 12x1+12x2+9x3+15x4+90x5+26x6+112x7
    s.t.  3x1+4x2+3x3+3x4+15x5+13x6+16x7 <= 35 ,  xi ∈ {0,1}

(α) Branch & Bound με στρατηγική BEST-FIRST (jumptracking): από όλους τους
    ζωντανούς κόμβους διακλαδίζεται κάθε φορά αυτός με το ΜΕΓΑΛΥΤΕΡΟ άνω
    φράγμα zLP. Το χαλαρωμένο LP λύνεται με τον «άπληστο» αλγόριθμο κατά
    φθίνοντα λόγο p_i/w_i (βέλτιστος για το κλασματικό knapsack).
(β) Δυϊκό του χαλαρωμένου LP + σχέσεις συμπληρωματικής χαλαρότητας και
    παραγωγή της δυϊκής λύσης από αυτές.
(γ) Ακέραιο δυϊκό με solver και έλεγχος των σχέσεων CS για το ΑΚΕΡΑΙΟ ζεύγος.
"""
from fractions import Fraction as F
import pulp

p = [12, 12, 9, 15, 90, 26, 112]
w = [3, 4, 3, 3, 15, 13, 16]
CAP = 35
n = 7
order = sorted(range(n), key=lambda i: (-F(p[i], w[i]), i))

print("Δεδομένα (ταξινομημένα κατά φθίνοντα λόγο p_i/w_i):")
print(f"  {'item':>6}{'p':>6}{'w':>5}{'p/w':>8}")
for i in order:
    print(f"  {'x'+str(i+1):>6}{p[i]:>6}{w[i]:>5}{float(F(p[i],w[i])):>8.3f}")


def lp_bound(fix):
    """Ακριβής λύση του χαλαρωμένου (κλασματικού) knapsack με άπληστο κανόνα."""
    cap, val = F(CAP), F(0)
    x = [F(0)] * n
    for i, vv in fix.items():
        x[i] = F(vv)
        cap -= w[i] * x[i]
        val += p[i] * x[i]
    if cap < 0:
        return None, None
    for i in order:
        if i in fix:
            continue
        take = min(F(1), cap / w[i])
        x[i] = take
        cap -= w[i] * take
        val += p[i] * take
    return x, val


def fractional(x):
    return [i for i in range(n) if x[i] not in (0, 1)]


# ======================================================================
# (α) BEST-FIRST BRANCH & BOUND
# ======================================================================
print("\n" + "=" * 94)
print("(α) BRANCH & BOUND - BEST FIRST")
print("=" * 94)
x0, z0 = lp_bound({})
nid = 1
rows = [(1, "ρίζα (χωρίς δεσμεύσεις)", None, x0, z0, "")]
# Λίστα ΖΩΝΤΑΝΩΝ κόμβων: (zLP, δεσμεύσεις, id). Best-first σημαίνει ότι
# επιλέγουμε κάθε φορά τον κόμβο με το ΜΕΓΑΛΥΤΕΡΟ άνω φράγμα zLP.
live = [(z0, {}, 1)]
incumbent, best_x, step = F(0), None, 0
seq = []
while live:
    best_node = max(live, key=lambda n: n[0])     # <-- best-first / jumptracking
    live.remove(best_node)
    zb, fix, myid = best_node
    step += 1
    if zb <= incumbent:
        seq.append((step, myid, f"αποκοπή από φράγμα ({zb} <= {incumbent})"))
        for r in range(len(rows)):
            if rows[r][0] == myid:
                rows[r] = rows[r][:5] + (f"αποκοπή από φράγμα ({zb} <= incumbent {incumbent})",)
        continue
    x, _ = lp_bound(fix)
    fr = fractional(x)
    j = fr[0]
    seq.append((step, myid, f"διακλάδωση στο x{j+1} (zLP = {zb})"))
    for r in range(len(rows)):
        if rows[r][0] == myid:
            rows[r] = rows[r][:5] + (f"διακλάδωση στο x{j+1}",)
    for val in (0, 1):
        nid += 1
        f2 = dict(fix)
        f2[j] = val
        x2, z2 = lp_bound(f2)
        if x2 is None:
            rows.append((nid, f"x{j+1}={val}", myid, None, None, "ΜΗ ΕΦΙΚΤΟ"))
            continue
        if z2 <= incumbent:
            act = f"αποκοπή από φράγμα ({z2} <= incumbent {incumbent})"
        elif not fractional(x2):
            incumbent, best_x = z2, x2
            act = f"ΑΚΕΡΑΙΟ  z = {z2}  ->  νέο incumbent"
        else:
            act = "ζωντανός"
            live.append((z2, f2, nid))
        rows.append((nid, f"x{j+1}={val}", myid, x2, z2, act))

print(f"\n{'κ':>3} {'γονέας':>7} {'δέσμευση':<10} {'λύση LP':<34} {'zLP':>6}  ενέργεια")
for r in rows:
    xs = "(" + ",".join(str(v) for v in r[3]) + ")" if r[3] else "-"
    zs = str(r[4]) if r[4] is not None else "-"
    print(f"{r[0]:>3} {str(r[2]):>7} {r[1]:<10} {xs:<34} {zs:>6}  {r[5]}")

print("\nΣειρά επιλογής κόμβων (best-first):")
for s, i, a in seq:
    print(f"  βήμα {s}: κόμβος κ{i} -> {a}")

sel = [i + 1 for i in range(n) if best_x[i] == 1]
print(f"\nΒΕΛΤΙΣΤΗ ΑΚΕΡΑΙΑ ΛΥΣΗ: x* = ({','.join(str(v) for v in best_x)})")
print(f"  αντικείμενα: {sel},  βάρος = {sum(w[i-1] for i in sel)} <= {CAP},  z* = {incumbent}")
print(f"  άνω φράγμα ρίζας zLP = {z0}  ->  κενό ακεραιότητας = {z0 - incumbent}")
print(f"  συνολικοί κόμβοι: {len(rows)}")

# ======================================================================
# (β) ΔΥΪΚΟ ΤΟΥ ΧΑΛΑΡΩΜΕΝΟΥ ΚΑΙ CS
# ======================================================================
print("\n" + "=" * 94)
print("(β) ΔΥΪΚΟ ΤΟΥ ΧΑΛΑΡΩΜΕΝΟΥ LP ΚΑΙ ΣΥΜΠΛΗΡΩΜΑΤΙΚΗ ΧΑΛΑΡΟΤΗΤΑ")
print("=" * 94)
print("  Πρωτεύον (χαλαρωμένο):  max Σ p_i x_i,  Σ w_i x_i <= 35 (u),  x_i <= 1 (v_i), x_i >= 0")
print("  Δυϊκό: min W = 35u + Σ v_i  s.t.  w_i u + v_i >= p_i ,  u, v_i >= 0")
print("\n  Σχέσεις CS:")
print("    (1) u > 0            =>  Σ w_i x_i = 35")
print("    (2) v_i > 0          =>  x_i = 1")
print("    (3) x_i > 0          =>  w_i u + v_i = p_i")
print("    (4) w_i u + v_i > p_i=>  x_i = 0")
print("    (5) x_i < 1          =>  v_i = 0")

xr = x0
print(f"\n  Από τη λύση του χαλαρωμένου x_LP = ({','.join(str(v) for v in xr)}):")
frac_i = fractional(xr)[0]
u = F(p[frac_i], w[frac_i])
print(f"    x{frac_i+1} = {xr[frac_i]} κλασματικό  =>  v{frac_i+1} = 0 (από (5)) ΚΑΙ "
      f"w{frac_i+1} u = p{frac_i+1} (από (3))  =>  u* = {p[frac_i]}/{w[frac_i]} = {u}")
v = []
for i in range(n):
    vi = max(F(0), p[i] - w[i] * u)
    v.append(vi)
    if xr[i] == 1:
        print(f"    x{i+1} = 1  =>  {w[i]}*{u} + v{i+1} = {p[i]}  =>  v{i+1} = {vi}")
    elif xr[i] == 0:
        print(f"    x{i+1} = 0  και w{i+1}u = {w[i]*u} >= p{i+1} = {p[i]}  =>  v{i+1} = 0")
W = CAP * u + sum(v)
print(f"\n  ΔΥΪΚΗ ΛΥΣΗ: u* = {u},  v* = ({','.join(str(t) for t in v)})")
print(f"  W* = 35*{u} + {sum(v)} = {W}   =   zLP = {z0}   ->  ισχυρή δυϊκότητα ✓")

# ======================================================================
# (γ) ΑΚΕΡΑΙΟ ΔΥΪΚΟ ΚΑΙ ΕΛΕΓΧΟΣ CS ΓΙΑ ΤΟ ΑΚΕΡΑΙΟ ΖΕΥΓΟΣ
# ======================================================================
print("\n" + "=" * 94)
print("(γ) ΑΚΕΡΑΙΟ ΔΥΪΚΟ ΚΑΙ ΕΛΕΓΧΟΣ CS")
print("=" * 94)
d = pulp.LpProblem("int_dual", pulp.LpMinimize)
U = pulp.LpVariable("u", lowBound=0, cat="Integer")
V = [pulp.LpVariable(f"v{i+1}", lowBound=0, cat="Integer") for i in range(n)]
d += CAP * U + pulp.lpSum(V)
for i in range(n):
    d += w[i] * U + V[i] >= p[i]
d.solve(pulp.PULP_CBC_CMD(msg=0))
ui = int(U.value())
vi = [int(t.value()) for t in V]
Wi = CAP * ui + sum(vi)
print(f"  u* = {ui},  v* = {vi},  W*_int = {Wi}")

print(f"\n  {'i':>3}{'w_i u':>8}{'v_i':>6}{'άθροισμα':>10}{'p_i':>6}{'περίσσεια':>11}{'x_i*':>6}")
for i in range(n):
    tot = w[i] * ui + vi[i]
    print(f"  {i+1:>3}{w[i]*ui:>8}{vi[i]:>6}{tot:>10}{p[i]:>6}{tot-p[i]:>11}{int(best_x[i]):>6}")

load = sum(w[i] for i in range(n) if best_x[i] == 1)
print(f"\n  Έλεγχος (3): x_i = 1  =>  w_i u + v_i = p_i :  "
      f"{all(w[i]*ui+vi[i] == p[i] for i in range(n) if best_x[i] == 1)}  ✓")
print(f"  Έλεγχος (2): v_i > 0  =>  x_i = 1 :  "
      f"{all(best_x[i] == 1 for i in range(n) if vi[i] > 0)}  ✓")
print(f"  Έλεγχος (1): u = {ui} > 0  =>  Σ w_i x_i = 35 ;  στην πραγματικότητα "
      f"Σ w_i x_i = {load} < 35  ->  Η CS ΠΑΡΑΒΙΑΖΕΤΑΙ")
print(f"\n  Ερμηνεία: οι σχέσεις CS είναι ισοδύναμες με τη βελτιστότητα ΜΟΝΟ για το")
print(f"  ΣΥΝΕΧΕΣ ζεύγος πρωτεύοντος-δυϊκού. Εδώ z*_int = {incumbent} < {Wi} = W*_int,")
print(f"  δηλ. υπάρχει κενό ακεραιότητας (integrality gap) = {Wi - incumbent},")
print(f"  οπότε δεν μπορεί να ισχύουν όλες οι σχέσεις CS ταυτόχρονα.")

# επαλήθευση της ακέραιας πρωτεύουσας λύσης με solver
q = pulp.LpProblem("knap", pulp.LpMaximize)
X = [pulp.LpVariable(f"x{i+1}", cat="Binary") for i in range(n)]
q += pulp.lpSum(p[i] * X[i] for i in range(n))
q += pulp.lpSum(w[i] * X[i] for i in range(n)) <= CAP
q.solve(pulp.PULP_CBC_CMD(msg=0))
print(f"\n  [επαλήθευση] ακέραιο πρωτεύον με CBC: "
      f"x = ({','.join(str(int(t.value())) for t in X)}), z = {pulp.value(q.objective)}")
