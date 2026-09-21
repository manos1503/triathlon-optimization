"""
Εργασία 2 - Άσκηση 5
Branch & Bound για το μικτό ακέραιο πρόβλημα

    min z = 12x1 + 45x2 + 67x3 + x4
    (P1) 4x1 + 2x2 - x4 <= 10
    (P2) 6x1 + 19x3     >=  5
         x2, x3, x4 >= 0 ,  x1 ∈ {0,1} ,  x3 ακέραιος

Ο κώδικας υλοποιεί ρητά το B&B (δεν το αφήνει στον MILP solver): σε κάθε
κόμβο λύνει μόνο το ΧΑΛΑΡΩΜΕΝΟ LP με PuLP/CBC και εφαρμόζει τα τρία
κριτήρια αποκοπής (fathoming):
  (i)   μη εφικτότητα,
  (ii)  ακέραια λύση -> ενημέρωση incumbent,
  (iii) φράγμα: zLP >= incumbent  (πρόβλημα ΕΛΑΧΙΣΤΟΠΟΙΗΣΗΣ)
Η επιλογή του επόμενου κόμβου γίνεται με best-first (μικρότερο zLP).
"""
import pulp

INT_VARS = ["x1", "x3"]          # ακέραιες μεταβλητές
BOUNDS0 = {"x1": (0, 1), "x2": (0, None), "x3": (0, None), "x4": (0, None)}


def solve_lp(bnds):
    """Χαλαρωμένο LP του κόμβου· επιστρέφει (status, x, z)."""
    p = pulp.LpProblem("node", pulp.LpMinimize)
    v = {k: pulp.LpVariable(k, lowBound=bnds[k][0], upBound=bnds[k][1])
         for k in BOUNDS0}
    p += 12 * v["x1"] + 45 * v["x2"] + 67 * v["x3"] + v["x4"]
    p += 4 * v["x1"] + 2 * v["x2"] - v["x4"] <= 10
    p += 6 * v["x1"] + 19 * v["x3"] >= 5
    st = p.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[st] != "Optimal":
        return pulp.LpStatus[st], None, None
    return "Optimal", {k: v[k].value() for k in BOUNDS0}, pulp.value(p.objective)


def frac_var(x):
    for k in INT_VARS:
        if abs(x[k] - round(x[k])) > 1e-6:
            return k
    return None


counter = 0
incumbent, best_x = float("inf"), None
st, x, z = solve_lp(dict(BOUNDS0))
counter += 1
nodes = [(1, "ρίζα", None, st, x, z, "")]
# Λίστα ΖΩΝΤΑΝΩΝ κόμβων: (zLP, όρια, id). Η επιλογή γίνεται με best-first,
# δηλαδή παίρνουμε κάθε φορά τον κόμβο με το ΜΙΚΡΟΤΕΡΟ zLP (ελαχιστοποίηση).
live = [(z, dict(BOUNDS0), 1)]
nid = 1
print("=" * 90)
print("BRANCH & BOUND (best-first, ελαχιστοποίηση)")
print("=" * 90)
print(f"κ1 (ρίζα): LP -> x = {x}, zLP = {z:.4f}")

while live:
    best_node = min(live, key=lambda n: n[0])     # <-- best-first
    live.remove(best_node)
    z, bnds, myid = best_node
    if z >= incumbent - 1e-9:
        print(f"  κ{myid}: ΑΠΟΚΟΠΗ από φράγμα (zLP = {z:.4f} >= incumbent = {incumbent:.4f})")
        continue
    st, x, _ = solve_lp(bnds)
    j = frac_var(x)
    print(f"\n  Διακλάδωση του κ{myid} (zLP = {z:.4f}) στη μεταβλητή {j} = {x[j]:.4f}")
    lo, hi = bnds[j]
    for side, nb in (("<= " + str(int(x[j] // 1)), (lo, int(x[j] // 1))),
                     (">= " + str(int(x[j] // 1) + 1), (int(x[j] // 1) + 1, hi))):
        nid += 1
        child = dict(bnds)
        child[j] = nb
        if nb[1] is not None and nb[0] > nb[1]:
            print(f"    κ{nid}: {j} {side}  ->  ΚΕΝΟ ΔΙΑΣΤΗΜΑ (μη εφικτό)")
            nodes.append((nid, f"{j} {side}", myid, "Infeasible", None, None, "μη εφικτό"))
            continue
        s2, x2, z2 = solve_lp(child)
        counter += 1
        if s2 != "Optimal":
            print(f"    κ{nid}: {j} {side}  ->  ΜΗ ΕΦΙΚΤΟ (αποκοπή)")
            nodes.append((nid, f"{j} {side}", myid, "Infeasible", None, None, "μη εφικτό"))
            continue
        f2 = frac_var(x2)
        if z2 >= incumbent - 1e-9:
            act = f"αποκοπή από φράγμα ({z2:.2f} >= {incumbent:.2f})"
        elif f2 is None:
            incumbent, best_x = z2, x2
            act = f"ΑΚΕΡΑΙΑ -> νέο incumbent z = {z2:.2f}"
        else:
            act = f"ζωντανός (κλασματικό {f2})"
            live.append((z2, child, nid))
        print(f"    κ{nid}: {j} {side}  ->  x = "
              f"{ {k: round(vv,4) for k,vv in x2.items()} }, zLP = {z2:.4f}   [{act}]")
        nodes.append((nid, f"{j} {side}", myid, s2, x2, z2, act))

print("\n" + "=" * 90)
print(f"ΒΕΛΤΙΣΤΗ ΑΚΕΡΑΙΑ ΛΥΣΗ: x* = { {k: round(v,4) for k,v in best_x.items()} }")
print(f"z* = {incumbent:.4f}   (LP φράγμα ρίζας = {nodes[0][5]:.4f}, "
      f"κενό ακεραιότητας = {incumbent - nodes[0][5]:.4f})")
print(f"Συνολικοί κόμβοι που λύθηκαν: {counter}")

# ---------------------------------------------------- ανεξάρτητη επαλήθευση
p = pulp.LpProblem("check", pulp.LpMinimize)
x1 = pulp.LpVariable("x1", cat="Binary")
x2 = pulp.LpVariable("x2", lowBound=0)
x3 = pulp.LpVariable("x3", lowBound=0, cat="Integer")
x4 = pulp.LpVariable("x4", lowBound=0)
p += 12 * x1 + 45 * x2 + 67 * x3 + x4
p += 4 * x1 + 2 * x2 - x4 <= 10
p += 6 * x1 + 19 * x3 >= 5
p.solve(pulp.PULP_CBC_CMD(msg=0))
print(f"\n[επαλήθευση MILP solver] x = ({x1.value()}, {x2.value()}, {x3.value()}, "
      f"{x4.value()}), z = {pulp.value(p.objective)}")
