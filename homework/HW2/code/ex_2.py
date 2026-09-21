"""
Εργασία 2 - Άσκηση 2
Ανάθεση 4 εργασιών (Α,Β,Γ,Δ) σε 3 εργαστήρια με δυνατότητα κλασματικής
εκτέλεσης, με ελάχιστο συνολικό κόστος.

Μεταβλητές:  x_ij = ποσοστό (κλάσμα) της εργασίας j στο εργαστήριο i
Κόστος:      c_ij = t_ij * k_i   (ώρες * κόστος/ώρα)

    min   Σ_i Σ_j c_ij x_ij
    s.t.  Σ_i x_ij = 1            για κάθε εργασία j     (πλήρης εκτέλεση)
          Σ_j t_ij x_ij <= 160    για κάθε εργαστήριο i  (διαθέσιμος χρόνος)
          0 <= x_ij <= 1

Επιπλέον υπολογίζονται οι δυϊκές τιμές (σκιώδεις τιμές του χρόνου κάθε
εργαστηρίου και "τιμή" κάθε εργασίας) και επαληθεύεται η ισχυρή δυϊκότητα.
"""
import pulp

LABS = [1, 2, 3]
JOBS = ["A", "B", "Γ", "Δ"]
t = {(1, "A"): 32, (1, "B"): 151, (1, "Γ"): 72, (1, "Δ"): 118,
     (2, "A"): 39, (2, "B"): 147, (2, "Γ"): 61, (2, "Δ"): 126,
     (3, "A"): 46, (3, "B"): 155, (3, "Γ"): 57, (3, "Δ"): 121}
k = {1: 89, 2: 81, 3: 84}
CAP = {i: 160 for i in LABS}
c = {(i, j): t[i, j] * k[i] for i in LABS for j in JOBS}

print("Πίνακας κόστους c_ij = t_ij * k_i (ευρώ για ΟΛΟΚΛΗΡΗ την εργασία):")
print("        " + "".join(f"{j:>10}" for j in JOBS))
for i in LABS:
    print(f" Εργ.{i}  " + "".join(f"{c[i,j]:>10}" for j in JOBS))

# ------------------------------------------------------------- μοντέλο
prob = pulp.LpProblem("anathesi_ergasiwn", pulp.LpMinimize)
x = pulp.LpVariable.dicts("x", (LABS, JOBS), lowBound=0, upBound=1, cat="Continuous")
prob += pulp.lpSum(c[i, j] * x[i][j] for i in LABS for j in JOBS), "Synoliko_Kostos"
for j in JOBS:
    prob += pulp.lpSum(x[i][j] for i in LABS) == 1, f"ekteleshb_{j}"
for i in LABS:
    prob += pulp.lpSum(t[i, j] * x[i][j] for j in JOBS) <= CAP[i], f"xronos_{i}"

prob.solve(pulp.PULP_CBC_CMD(msg=0))
print(f"\nΚατάσταση: {pulp.LpStatus[prob.status]}")
print(f"Ελάχιστο συνολικό κόστος Z* = {pulp.value(prob.objective):.4f} ευρώ\n")

print("Βέλτιστη ανάθεση x_ij:")
print("        " + "".join(f"{j:>10}" for j in JOBS) + f"{'ώρες':>10}")
for i in LABS:
    hrs = sum(t[i, j] * x[i][j].value() for j in JOBS)
    print(f" Εργ.{i}  " + "".join(f"{x[i][j].value():>10.6f}" for j in JOBS)
          + f"{hrs:>10.4f}")

print("\nΔυϊκές τιμές (σκιώδεις τιμές):")
for name, cons in prob.constraints.items():
    print(f"  {name:<14} dual = {cons.pi:>12.4f}   slack = {cons.slack}")

# --------------------------------------------- έλεγχος ισχυρής δυϊκότητας
u = {j: prob.constraints[f"ekteleshb_{j}"].pi for j in JOBS}   # ελεύθερες
w = {i: prob.constraints[f"xronos_{i}"].pi for i in LABS}      # <= 0 (min)
W = sum(u[j] for j in JOBS) + sum(CAP[i] * w[i] for i in LABS)
print(f"\nΤιμή δυϊκής αντικειμενικής = {W:.4f}   (ισούται με Z* -> ισχυρή δυϊκότητα)")

# ----------------------------- σύγκριση με «ακέραια» ανάθεση (κάθε εργασία
# ολόκληρη σε ένα εργαστήριο) ώστε να φανεί η αξία της κλασματικής εκτέλεσης
probI = pulp.LpProblem("anathesi_akeraia", pulp.LpMinimize)
y = pulp.LpVariable.dicts("y", (LABS, JOBS), cat="Binary")
probI += pulp.lpSum(c[i, j] * y[i][j] for i in LABS for j in JOBS)
for j in JOBS:
    probI += pulp.lpSum(y[i][j] for i in LABS) == 1
for i in LABS:
    probI += pulp.lpSum(t[i, j] * y[i][j] for j in JOBS) <= CAP[i]
probI.solve(pulp.PULP_CBC_CMD(msg=0))
print(f"\nΓια σύγκριση - αν ΔΕΝ επιτρεπόταν κλασματική εκτέλεση:")
print(f"  κατάσταση: {pulp.LpStatus[probI.status]}, κόστος = "
      f"{pulp.value(probI.objective):.2f} ευρώ  "
      f"(διαφορά +{pulp.value(probI.objective)-pulp.value(prob.objective):.2f})")
for j in JOBS:
    for i in LABS:
        if y[i][j].value() > 0.5:
            print(f"    εργασία {j} -> Εργαστήριο {i}")
