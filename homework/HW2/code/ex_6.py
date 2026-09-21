"""
Εργασία 2 - Άσκηση 6
Διατύπωση ως ΜΙΚΤΟ ΑΚΕΡΑΙΟ πρόγραμμα και επίλυση του

    min f1(x1) + f2(x2),   f1 = 5+4x1 (x1>0) ή 0 (x1=0)
                           f2 = 3+5x2 (x2>0) ή 0 (x2=0)
    1) x1 >= 5  Ή  x2 >= 5
    2) τουλάχιστον ένας από: 2x1+x2>=12 , x1+x2>=10 , x1+2x2>=12
    3) |x1 - x2| ∈ {0, 5, 10}
    4) x1, x2 >= 0

ΤΕΧΝΙΚΕΣ:
  * fixed cost      : x_i <= M y_i , όρος +F_i y_i στην αντικειμενική
  * διάζευξη (OR)   : z1+z2 >= 1 , x1 >= 5 z1 , x2 >= 5 z2
  * at-least-one    : w1+w2+w3 >= 1 και g_r(x) >= b_r - M(1-w_r)
  * |x1-x2| = v     : v = 5 a5 + 10 a10 με a0+a5+a10 = 1 και δυαδική s
                      για το πρόσημο (x1-x2 = v αν s=1, x2-x1 = v αν s=0)

ΤΕΚΜΗΡΙΩΣΗ ΤΟΥ M: κάθε βέλτιστη λύση έχει x1, x2 <= 10 (βλ. ανάλυση
περιπτώσεων παρακάτω), άρα το φράγμα x_i <= 20 δεν αποκόπτει καμία βέλτιστη
λύση και M = 40 είναι ασφαλές (>= 2*20 ώστε να «σβήνει» κάθε ανενεργό
περιορισμό). Ο κώδικας επαληθεύει ότι το αποτέλεσμα ΔΕΝ εξαρτάται από το M.
"""
import pulp

F1, V1 = 5, 4          # σταθερό και μεταβλητό κόστος της f1
F2, V2 = 3, 5          # σταθερό και μεταβλητό κόστος της f2
UB = 20                # ασφαλές άνω φράγμα για τα x1, x2


def build(M, force_v=None):
    p = pulp.LpProblem("mip", pulp.LpMinimize)
    x1 = pulp.LpVariable("x1", 0, UB)
    x2 = pulp.LpVariable("x2", 0, UB)
    y1 = pulp.LpVariable("y1", cat="Binary")
    y2 = pulp.LpVariable("y2", cat="Binary")
    z1 = pulp.LpVariable("z1", cat="Binary")
    z2 = pulp.LpVariable("z2", cat="Binary")
    w = [pulp.LpVariable(f"w{r}", cat="Binary") for r in range(3)]
    a0 = pulp.LpVariable("a0", cat="Binary")
    a5 = pulp.LpVariable("a5", cat="Binary")
    a10 = pulp.LpVariable("a10", cat="Binary")
    s = pulp.LpVariable("s", cat="Binary")

    p += F1 * y1 + V1 * x1 + F2 * y2 + V2 * x2          # (γραμμικοποιημένη)

    p += x1 <= UB * y1                                   # fixed costs
    p += x2 <= UB * y2

    p += z1 + z2 >= 1                                    # περιορισμός 1 (OR)
    p += x1 >= 5 * z1
    p += x2 >= 5 * z2

    p += w[0] + w[1] + w[2] >= 1                         # περιορισμός 2
    p += 2 * x1 + x2 >= 12 - M * (1 - w[0])
    p += x1 + x2 >= 10 - M * (1 - w[1])
    p += x1 + 2 * x2 >= 12 - M * (1 - w[2])

    p += a0 + a5 + a10 == 1                              # περιορισμός 3
    v = 5 * a5 + 10 * a10
    if force_v is not None:                              # για την ανάλυση
        p += a0 == (1 if force_v == 0 else 0)
        p += a5 == (1 if force_v == 5 else 0)
        p += a10 == (1 if force_v == 10 else 0)
    p += x1 - x2 <= v + M * (1 - s)
    p += x1 - x2 >= v - M * (1 - s)
    p += x2 - x1 <= v + M * s
    p += x2 - x1 >= v - M * s
    return p, x1, x2, (y1, y2, z1, z2, w, a0, a5, a10, s)


print("=== Επίλυση του πλήρους MIP ===")
p, X1, X2, B = build(40)
p.solve(pulp.PULP_CBC_CMD(msg=0))
y1, y2, z1, z2, w, a0, a5, a10, s = B
print(f" κατάσταση: {pulp.LpStatus[p.status]}")
print(f" x1* = {X1.value():.6f} (= 17/3),  x2* = {X2.value():.6f} (= 2/3)")
print(f" Z*  = {pulp.value(p.objective):.6f}")
print(f" δυαδικές: y1={y1.value():.0f} y2={y2.value():.0f} | z1={z1.value():.0f} "
      f"z2={z2.value():.0f} | w={[int(v.value()) for v in w]} | "
      f"a0={a0.value():.0f} a5={a5.value():.0f} a10={a10.value():.0f} | s={s.value():.0f}")

x1v, x2v = X1.value(), X2.value()
print("\n Έλεγχος των ΑΡΧΙΚΩΝ (μη γραμμικών) περιορισμών:")
print(f"  1) x1>=5 ή x2>=5 : {x1v:.4f} >= 5 -> {x1v >= 5 - 1e-6}")
print(f"  2) 2x1+x2 = {2*x1v+x2v:.4f} >= 12 -> {2*x1v+x2v >= 12-1e-6}"
      f" | x1+x2 = {x1v+x2v:.4f} | x1+2x2 = {x1v+2*x2v:.4f}")
print(f"  3) |x1-x2| = {abs(x1v-x2v):.4f} ∈ {{0,5,10}} -> {abs(abs(x1v-x2v)-5) < 1e-6}")
print(f"  4) x1,x2 >= 0 -> True")
f1 = F1 + V1 * x1v if x1v > 1e-9 else 0
f2 = F2 + V2 * x2v if x2v > 1e-9 else 0
print(f"  f1({x1v:.4f}) = {f1:.4f} ,  f2({x2v:.4f}) = {f2:.4f} ,  σύνολο = {f1+f2:.4f}")

print("\n=== Ανεξαρτησία από το M (ευρωστία της διατύπωσης) ===")
for M in (25, 40, 100, 1000):
    pp, a, b, _ = build(M)
    pp.solve(pulp.PULP_CBC_CMD(msg=0))
    print(f"  M = {M:>5}: Z* = {pulp.value(pp.objective):.6f}, "
          f"x = ({a.value():.4f}, {b.value():.4f})")

print("\n=== Ανάλυση ανά τιμή του |x1 - x2| ===")
for v in (0, 5, 10):
    pp, a, b, _ = build(40, force_v=v)
    pp.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[pp.status] == "Optimal":
        print(f"  |x1-x2| = {v:>2}:  x = ({a.value():.4f}, {b.value():.4f}),  "
              f"Z = {pulp.value(pp.objective):.4f}")
    else:
        print(f"  |x1-x2| = {v:>2}:  {pulp.LpStatus[pp.status]}")

print("\n=== Ανεξάρτητη επαλήθευση με πλήρη ανάλυση περιπτώσεων ===")
# Σε κάθε περίπτωση το πρόβλημα γίνεται 1-διάστατο και λύνεται αναλυτικά.
best = (float("inf"), None)
cases = []
# x2 = 0 : τότε πρέπει x1 >= 5 και |x1| ∈ {0,5,10} -> x1 ∈ {5, 10}
for x1v in (5, 10):
    ok2 = max(2*x1v, x1v, x1v) >= 12 if False else (2*x1v >= 12 or x1v >= 10 or x1v >= 12)
    if ok2:
        cases.append((x1v, 0, F1 + V1*x1v))
# x1 = 0 : x2 ∈ {5, 10}
for x2v in (5, 10):
    if (x2v >= 12) or (x2v >= 10) or (2*x2v >= 12):
        cases.append((0, x2v, F2 + V2*x2v))
# x1, x2 > 0 : για κάθε d ∈ {0,5,10}, κάθε πρόσημο και κάθε επιλογή του
# "τουλάχιστον ενός" περιορισμού της ομάδας 2, μένει ένα ΓΡΑΜΜΙΚΟ πρόβλημα
# μιας μεταβλητής, που λύνεται ακριβώς (Fraction).
from fractions import Fraction as Fr
G = [(2, 1, 12), (1, 1, 10), (1, 2, 12)]      # a*x1 + b*x2 >= c
for d in (0, 5, 10):
    for sgn in (+1, -1):
        for (a, b, cst) in G:
            # x1 = t + d αν sgn=+1 αλλιώς x2 = t + d, με t >= 0
            if sgn == 1:
                # a(t+d) + b t >= cst  ->  t >= (cst - a d)/(a+b)
                t = max(Fr(0), Fr(cst - a*d, a + b))
                # και ο περιορισμός 1: x1 = t+d >= 5  ή  x2 = t >= 5
                if not (t + d >= 5):
                    t = max(t, Fr(5))
                x1v, x2v = t + d, t
            else:
                t = max(Fr(0), Fr(cst - b*d, a + b))
                if not (t + d >= 5):
                    t = max(t, Fr(5))
                x1v, x2v = t, t + d
            if x1v <= 0 or x2v <= 0:
                continue
            cases.append((float(x1v), float(x2v),
                          float(F1 + V1*x1v + F2 + V2*x2v)))
for x1v, x2v, val in cases:
    if val < best[0] - 1e-9:
        best = (val, (x1v, x2v))
print(f"  ελάχιστο από την ανάλυση περιπτώσεων: Z = {best[0]:.4f} στο "
      f"x = ({best[1][0]:.4f}, {best[1][1]:.4f})")
