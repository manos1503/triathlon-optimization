"""
Εργασία 1 - Άσκηση 2
Μοντελοποίηση και γραφική επίλυση του προβλήματος βέλτιστου χαρτοφυλακίου.

    max Z = 0.11 xD + 0.17 xF
    (P1) xD + xF <= 12      (συνολικό κεφάλαιο)
    (P2) xD      <= 10      (όριο εγχώριων)
    (P3)      xF <=  7      (όριο ξένων)
    (P4) -0.5 xD + xF >= 0  (ξένες >= μισό των εγχώριων)
    (P5) xD - 0.5 xF  >= 0  (εγχώριες >= μισό των ξένων)
         xD, xF >= 0

Ο κώδικας απαριθμεί όλες τις τομές ζευγών ευθειών, ξεχωρίζει τις κορυφές,
ελέγχει ποιοι περιορισμοί είναι πλεονάζοντες (redundant), σχεδιάζει την
εφικτή περιοχή και επαληθεύει με scipy.linprog. Επιπλέον υπολογίζει τις
σκιώδεις τιμές (dual values) των δεσμευτικών περιορισμών.
"""
import itertools
from fractions import Fraction as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

# a*xD + b*xF (sense) c
LINES = [
    (F(1),      F(1),  F(12), "<=", "P1: xD+xF<=12"),
    (F(1),      F(0),  F(10), "<=", "P2: xD<=10"),
    (F(0),      F(1),  F(7),  "<=", "P3: xF<=7"),
    (F(-1, 2),  F(1),  F(0),  ">=", "P4: xF>=0.5xD"),
    (F(1),  F(-1, 2),  F(0),  ">=", "P5: xD>=0.5xF"),
]
C = (F(11, 100), F(17, 100))


def feasible(p):
    u, v = p
    if u < 0 or v < 0:
        return False
    for a, b, c, s, _ in LINES:
        lhs = a * u + b * v
        if (s == "<=" and lhs > c) or (s == ">=" and lhs < c):
            return False
    return True


# ---- 1. όλες οι τομές ζευγών ευθειών ------------------------------------
cand = {}
for (a1, b1, c1, _, n1), (a2, b2, c2, _, n2) in itertools.combinations(LINES, 2):
    det = a1 * b2 - a2 * b1
    if det == 0:
        continue
    p = ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)
    cand.setdefault(p, set()).update([n1, n2])

verts = sorted([p for p in cand if feasible(p)], key=lambda q: float(C[0]*q[0]+C[1]*q[1]))
print("Κορυφές της εφικτής περιοχής:")
for p in verts:
    act = [nm for a, b, c, s, nm in LINES if a*p[0] + b*p[1] == c]
    z = C[0]*p[0] + C[1]*p[1]
    print(f"  (xD,xF)=({str(p[0]):>5},{str(p[1]):>5})  Z={float(z):8.4f} εκατ."
          f"  ενεργοί: {act}")
best = verts[-1]
print(f"\nΒΕΛΤΙΣΤΗ: xD*={best[0]}, xF*={best[1]}, Z*={float(C[0]*best[0]+C[1]*best[1]):.4f} εκατ.")

# ---- 2. έλεγχος πλεοναζόντων (redundant) περιορισμών ---------------------
# Ένας περιορισμός είναι πλεονάζων όταν ΔΕΝ μπορεί να παραβιαστεί από κανένα
# σημείο που ικανοποιεί όλους τους υπόλοιπους: βελτιστοποιούμε το αριστερό
# του μέλος πάνω στην περιοχή των υπολοίπων και συγκρίνουμε με το b.
print("\nΈλεγχος πλεοναζόντων (redundant) περιορισμών:")
for k, (a, b, c, s, nm) in enumerate(LINES):
    rest = LINES[:k] + LINES[k+1:]
    A_ub, b_ub = [], []
    for aa, bb, cc, ss, _ in rest:
        A_ub.append([float(aa), float(bb)] if ss == "<=" else [-float(aa), -float(bb)])
        b_ub.append(float(cc) if ss == "<=" else -float(cc))
    # <= : μεγιστοποιούμε το LHS · >= : ελαχιστοποιούμε το LHS
    obj = [-float(a), -float(b)] if s == "<=" else [float(a), float(b)]
    r = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*2)
    if r.status != 0:                       # μη φραγμένο -> μπορεί να παραβιαστεί
        print(f"  {nm:<18}: ΟΧΙ πλεονάζων (το LHS δεν είναι φραγμένο)")
        continue
    ext = -r.fun if s == "<=" else r.fun
    red = (ext <= float(c) + 1e-9) if s == "<=" else (ext >= float(c) - 1e-9)
    print(f"  {nm:<18}: ακρότατο LHS στους υπόλοιπους = {ext:8.4f} vs b = {float(c):6.2f}"
          f"  ->  {'ΠΛΕΟΝΑΖΩΝ' if red else 'ΟΧΙ πλεονάζων'}")

# ---- 3. επαλήθευση + σκιώδεις τιμές -------------------------------------
res = linprog([-0.11, -0.17],
              A_ub=[[1, 1], [1, 0], [0, 1], [0.5, -1], [-1, 0.5]],
              b_ub=[12, 10, 7, 0, 0], bounds=[(0, None)]*2)
print(f"\nscipy.linprog: x={np.round(res.x,6)}  Z={-res.fun:.4f}")

# Οι σκιώδεις τιμές προκύπτουν από τους ΔΕΣΜΕΥΤΙΚΟΥΣ περιορισμούς: αν B είναι
# ο πίνακας με γραμμές τα διανύσματα των δεσμευτικών περιορισμών, τότε το y
# λύνει το B^T y = c  (συνθήκη βελτιστότητας στη βέλτιστη κορυφή).
binding = [(nm, [float(a), float(b)]) for a, b, c, s, nm in LINES
           if a*best[0] + b*best[1] == c]
B = np.array([row for _, row in binding])
y = np.linalg.solve(B.T, np.array([0.11, 0.17]))
print("\nΣκιώδεις τιμές (μόνο οι δεσμευτικοί έχουν μη μηδενική τιμή):")
for (nm, _), yi in zip(binding, y):
    print(f"  {nm:<18} y = {yi:.4f}")
for a, b, c, s, nm in LINES:
    if a*best[0] + b*best[1] != c:
        print(f"  {nm:<18} y = 0.0000   (μη δεσμευτικός)")

# ---- 4. σχήμα -----------------------------------------------------------
# Οι ετικέτες των κορυφών τοποθετούνται ΑΚΤΙΝΙΚΑ προς τα έξω (μακριά από το
# κέντρο βάρους του πολυγώνου) ώστε να μην αλληλεπικαλύπτονται, και η λεζάντα
# μπαίνει κάτω από τους άξονες.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
XL, YL = (0, 13), (0, 13)

fig, ax = plt.subplots(figsize=(7.4, 6.0))
g = np.linspace(XL[0], XL[1], 800)
UU, VV = np.meshgrid(g, g)
ok = (UU >= 0) & (VV >= 0)
for a, b, c, s, _ in LINES:
    lhs = float(a)*UU + float(b)*VV
    ok &= (lhs <= float(c) + 1e-9) if s == "<=" else (lhs >= float(c) - 1e-9)
ax.contourf(UU, VV, ok.astype(float), levels=[.5, 1.5], colors=["#9fd3e8"],
            alpha=.60, zorder=0)

for k, (a, b, c, s, nm) in enumerate(LINES):
    col = PALETTE[k % len(PALETTE)]
    a, b, c = float(a), float(b), float(c)
    if b != 0:
        ax.plot(g, (c - a*g)/b, lw=1.4, color=col, label=nm, zorder=2)
    else:
        ax.axvline(c/a, lw=1.4, color=col, label=nm, zorder=2)

P = np.array([[float(p[0]), float(p[1])] for p in verts])
ax.plot(P[:, 0], P[:, 1], "o", color="crimson", ms=7, zorder=5)

cx, cy = P[:, 0].mean(), P[:, 1].mean()
bx, by = float(best[0]), float(best[1])
gvec = np.array([0.11, 0.17]); gvec = gvec / np.hypot(*gvec)

for p in verts:
    px, py = float(p[0]), float(p[1])
    d = np.array([px - cx, py - cy])
    if np.hypot(*d) < 1e-9:
        d = np.array([-1.0, -1.0])
    d = d / np.hypot(*d)
    if abs(px - bx) < 1e-9 and abs(py - by) < 1e-9:
        d = -gvec                       # η ετικέτα αντίθετα από το βέλος
    # κορυφή κολλητά στο περιθώριο -> η ετικέτα γυρίζει προς τα μέσα
    if px < XL[0] + 0.12 * (XL[1] - XL[0]):
        d[0] = abs(d[0])
    elif px > XL[1] - 0.12 * (XL[1] - XL[0]):
        d[0] = -abs(d[0])
    if py < YL[0] + 0.12 * (YL[1] - YL[0]):
        d[1] = abs(d[1])
    elif py > YL[1] - 0.12 * (YL[1] - YL[0]):
        d[1] = -abs(d[1])
    ax.annotate(f"({px:.2f}, {py:.2f})\nZ={float(C[0]*p[0]+C[1]*p[1]):.3f}",
                (px, py), textcoords="offset points",
                xytext=(d[0]*20, d[1]*20), fontsize=8, zorder=8,
                ha="left" if d[0] >= 0 else "right",
                va="bottom" if d[1] >= 0 else "top",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.75",
                          alpha=0.88, lw=0.5))

ax.plot([bx], [by], "o", color="gold", mec="k", ms=13, zorder=6)
L = 1.9
ax.annotate("", xy=(bx + gvec[0]*L, by + gvec[1]*L), xytext=(bx, by),
            arrowprops=dict(arrowstyle="-|>", lw=2.6, color="darkorange",
                            mutation_scale=18), zorder=7)
ax.annotate(r"$\nabla Z$", (bx + gvec[0]*L, by + gvec[1]*L),
            textcoords="offset points", xytext=(5, 3), fontsize=9,
            color="darkorange", fontweight="bold", zorder=8)

ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel("xD  (εκατ. € σε εγχώριες)"); ax.set_ylabel("xF  (εκατ. € σε ξένες)")
ax.set_title("Βέλτιστο χαρτοφυλάκιο: max Z = 0.11 xD + 0.17 xF", pad=10)
ax.grid(alpha=.28, zorder=1)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.11),
          ncol=3, frameon=False)
fig.tight_layout(); fig.savefig("ex2_graph.png", dpi=170, bbox_inches="tight")
