"""
Εργασία 1 - Άσκηση 1
Γραφική επίλυση του ΠΓΠ

    max Z = 3x1 + 7x2 + 4x3 - 3x4
    (P1) x1 + 2x2 <= 6      (P2) x3 + x4 <= 6
    (P3) 4x1 + 5x2 <= 20    (P4) -2x1 + x2 <= 1
    (P5) 2x3 - x4 <= 4      (P6) x3 >= 2
    (P7) 1 <= x2 <= 2       (P8) x1, x4 >= 0

Κανένας περιορισμός δεν αναμειγνύει τα (x1,x2) με τα (x3,x4), άρα το
πρόβλημα αποσυντίθεται σε δύο ανεξάρτητα 2-διάστατα ΠΓΠ, καθένα από τα
οποία λύνεται γραφικά. Ο κώδικας:
  1. απαριθμεί ΟΛΕΣ τις τομές ζευγών ευθειών κάθε υποπροβλήματος,
  2. κρατά όσες είναι εφικτές (= κορυφές του πολυγώνου),
  3. μετράει πόσες ευθείες περνούν από κάθε κορυφή (εντοπισμός εκφυλισμού),
  4. σχεδιάζει την εφικτή περιοχή, τις κορυφές και το διάνυσμα κλίσης,
  5. επαληθεύει το συνολικό αποτέλεσμα με τη scipy.optimize.linprog.
"""
import itertools
from fractions import Fraction as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

TOL = F(0)


# ---------------------------------------------------------------- εργαλεία
def intersect(l1, l2):
    """Τομή δύο ευθειών a*u + b*v = c (ακριβής αριθμητική με κλάσματα)."""
    (a1, b1, c1), (a2, b2, c2) = l1[:3], l2[:3]
    det = a1 * b2 - a2 * b1
    if det == 0:                      # παράλληλες -> δεν ορίζεται κορυφή
        return None
    u = (c1 * b2 - c2 * b1) / det
    v = (a1 * c2 - a2 * c1) / det
    return (u, v)


def analyse(lines, obj, title, fname, xlabel, ylabel, xlim, ylim):
    """Πλήρης γραφική ανάλυση ενός 2-διάστατου ΠΓΠ.

    lines: λίστα (a, b, c, sense, name) που εκφράζει a*u + b*v (<=|>=|=) c
    obj  : (c1, c2) συντελεστές της max c1*u + c2*v
    """
    def feasible(p):
        u, v = p
        for a, b, c, sense, _ in lines:
            lhs = a * u + b * v
            if sense == "<=" and lhs > c + TOL:
                return False
            if sense == ">=" and lhs < c - TOL:
                return False
        return True

    # ---- 1. όλες οι τομές ζευγών ευθειών --------------------------------
    cand = {}
    for l1, l2 in itertools.combinations(lines, 2):
        p = intersect(l1, l2)
        if p is None:
            continue
        cand.setdefault(p, set()).update([l1[4], l2[4]])

    # ---- 2. ποιες είναι κορυφές του πολυγώνου ---------------------------
    verts = {p: names for p, names in cand.items() if feasible(p)}

    # ---- 3. ενεργοί περιορισμοί ανά κορυφή (έλεγχος εκφυλισμού) --------
    print(f"\n===== {title} =====")
    rows = []
    for p in sorted(verts, key=lambda q: (float(q[0]), float(q[1]))):
        u, v = p
        active = [nm for a, b, c, s, nm in lines if a * u + b * v == c]
        z = obj[0] * u + obj[1] * v
        rows.append((p, active, z))
        flag = "  <-- ΕΚΦΥΛΙΣΜΕΝΗ (>2 ευθείες)" if len(active) > 2 else ""
        print(f"  ({str(u):>6}, {str(v):>6})  Z={str(z):>8}  ενεργοί: "
              f"{len(active)} {active}{flag}")
    best = max(rows, key=lambda r: r[2])
    print(f"  ΒΕΛΤΙΣΤΗ ΚΟΡΥΦΗ: {tuple(str(t) for t in best[0])}  Z*={best[2]}")

    # ---- 4. σχεδίαση -----------------------------------------------------
    plot_region(lines, rows, best, obj, title, fname, xlabel, ylabel, xlim, ylim)
    return best


# ---------------------------------------------------------- σχεδίαση
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
           "#8c564b", "#17becf", "#e377c2", "#7f7f7f"]


def plot_region(lines, rows, best, obj, title, fname, xlabel, ylabel, xlim, ylim):
    """Σχεδίαση εφικτής περιοχής με ετικέτες που δεν αλληλεπικαλύπτονται.

    Οι ετικέτες των κορυφών τοποθετούνται ΑΚΤΙΝΙΚΑ προς τα έξω (μακριά από το
    κέντρο βάρους του πολυγώνου), ώστε να μην πέφτουν η μία πάνω στην άλλη ούτε
    μέσα στην εφικτή περιοχή. Η λεζάντα μπαίνει ΚΑΤΩ από τους άξονες.
    """
    fig, ax = plt.subplots(figsize=(7.4, 6.0))

    # εφικτή περιοχή (σκίαση)
    gx = np.linspace(xlim[0], xlim[1], 800)
    gy = np.linspace(ylim[0], ylim[1], 800)
    UU, VV = np.meshgrid(gx, gy)
    ok = np.ones_like(UU, dtype=bool)
    for a, b, c, s, _ in lines:
        lhs = float(a) * UU + float(b) * VV
        ok &= (lhs <= float(c) + 1e-9) if s == "<=" else (lhs >= float(c) - 1e-9)
    ax.contourf(UU, VV, ok.astype(float), levels=[0.5, 1.5],
                colors=["#9fd3e8"], alpha=0.60, zorder=0)

    # ευθείες περιορισμών - ρητά διαφορετικά χρώματα
    for k, (a, b, c, s, nm) in enumerate(lines):
        col = PALETTE[k % len(PALETTE)]
        a, b, c = float(a), float(b), float(c)
        if b != 0:
            ax.plot(gx, (c - a * gx) / b, lw=1.4, color=col, label=nm, zorder=2)
        else:
            ax.axvline(c / a, lw=1.4, color=col, label=nm, zorder=2)

    pts = np.array([[float(p[0]), float(p[1])] for p, _, _ in rows])
    ax.plot(pts[:, 0], pts[:, 1], "o", color="crimson", ms=7, zorder=5)

    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()       # κέντρο βάρους κορυφών
    sx = (xlim[1] - xlim[0]) or 1.0
    sy = (ylim[1] - ylim[0]) or 1.0
    bx, by = float(best[0][0]), float(best[0][1])
    gvec = np.array([float(obj[0]) / sx, float(obj[1]) / sy])
    gvec = gvec / (np.hypot(*gvec) or 1.0)

    for (p, active, z) in rows:
        px, py = float(p[0]), float(p[1])
        d = np.array([(px - cx) / sx, (py - cy) / sy])
        if np.hypot(*d) < 1e-9:
            d = np.array([1.0, 1.0])
        d = d / np.hypot(*d)
        # η ετικέτα της βέλτιστης κορυφής πάει αντίθετα από το διάνυσμα κλίσης,
        # ώστε να μην την καλύπτει το βέλος
        if abs(px - bx) < 1e-9 and abs(py - by) < 1e-9:
            d = -gvec
        # ...και αν η κορυφή είναι κολλητά στο περιθώριο, η ετικέτα γυρίζει
        # προς τα μέσα ώστε να μη βγει έξω από τους άξονες
        if px < xlim[0] + 0.12 * sx:
            d[0] = abs(d[0])
        elif px > xlim[1] - 0.12 * sx:
            d[0] = -abs(d[0])
        if py < ylim[0] + 0.12 * sy:
            d[1] = abs(d[1])
        elif py > ylim[1] - 0.12 * sy:
            d[1] = -abs(d[1])
        off = 17.0
        ax.annotate(
            f"({px:.2f}, {py:.2f}){'*' if len(active) > 2 else ''}\nZ={float(z):.2f}",
            (px, py), textcoords="offset points",
            xytext=(d[0] * off, d[1] * off), fontsize=8, zorder=8,
            ha="left" if d[0] >= 0 else "right",
            va="bottom" if d[1] >= 0 else "top",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.75",
                      alpha=0.88, lw=0.5))

    # βέλτιστη κορυφή + διάνυσμα κλίσης
    ax.plot([bx], [by], "o", color="gold", mec="k", ms=13, zorder=6)
    L = 0.13
    ax.annotate("", xy=(bx + gvec[0] * L * sx, by + gvec[1] * L * sy), xytext=(bx, by),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color="darkorange",
                                mutation_scale=18), zorder=7)
    ax.annotate(r"$\nabla Z$", (bx + gvec[0] * L * sx, by + gvec[1] * L * sy),
                textcoords="offset points", xytext=(4 * np.sign(gvec[0] or 1), 4),
                fontsize=9, color="darkorange", fontweight="bold", zorder=8,
                ha="left" if gvec[0] >= 0 else "right")

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(alpha=0.28, zorder=1)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(fname, dpi=170, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------- Υποπρόβλημα 1: (x1, x2)
L1 = [
    (F(1),  F(2),  F(6),  "<=", "P1: x1+2x2<=6"),
    (F(4),  F(5),  F(20), "<=", "P3: 4x1+5x2<=20"),
    (F(-2), F(1),  F(1),  "<=", "P4: -2x1+x2<=1"),
    (F(0),  F(1),  F(2),  "<=", "P7a: x2<=2"),
    (F(0),  F(1),  F(1),  ">=", "P7b: x2>=1"),
    (F(1),  F(0),  F(0),  ">=", "P8a: x1>=0"),
]
b1 = analyse(L1, (F(3), F(7)), "Υποπρόβλημα 1:  max Z1 = 3x1 + 7x2",
             "ex1_subproblem1.png", "x1", "x2", (-0.6, 4.6), (0.0, 2.8))

# --------------------------------------------- Υποπρόβλημα 2: (x3, x4)
L2 = [
    (F(1), F(1),  F(6), "<=", "P2: x3+x4<=6"),
    (F(2), F(-1), F(4), "<=", "P5: 2x3-x4<=4"),
    (F(1), F(0),  F(2), ">=", "P6: x3>=2"),
    (F(0), F(1),  F(0), ">=", "P8b: x4>=0"),
]
b2 = analyse(L2, (F(4), F(-3)), "Υποπρόβλημα 2:  max Z2 = 4x3 - 3x4",
             "ex1_subproblem2.png", "x3", "x4", (1.4, 4.4), (-0.6, 5.0))

# --------------------------------------------- Σύνθεση & επαλήθευση
x1s, x2s = b1[0]; x3s, x4s = b2[0]
Z = 3 * x1s + 7 * x2s + 4 * x3s - 3 * x4s
print("\n===== Συνολική λύση =====")
print(f"x* = ({x1s}, {x2s}, {x3s}, {x4s}),  Z* = {Z}")

res = linprog([-3, -7, -4, 3],
              A_ub=[[1, 2, 0, 0], [0, 0, 1, 1], [4, 5, 0, 0],
                    [-2, 1, 0, 0], [0, 0, 2, -1]],
              b_ub=[6, 6, 20, 1, 4],
              bounds=[(0, None), (1, 2), (2, None), (0, None)])
print(f"Επαλήθευση με scipy.linprog: x = {np.round(res.x, 6)}, Z = {-res.fun:.6f}")
