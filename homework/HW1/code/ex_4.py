"""
Εργασία 1 - Άσκηση 4
Το ζητούμενο είναι μοντελοποίηση· ο κώδικας εδώ υπάρχει ΜΟΝΟ για να
επιβεβαιώσει ότι τα δύο μοντέλα (α) και (β) είναι σωστά διατυπωμένα και
δίνουν το αναμενόμενο αποτέλεσμα σε ένα μικρό αριθμητικό στιγμιότυπο.

(α)  max  Σ_k g_k d_k - Σ_k Σ_r c_rk x_rk
     s.t. Σ_r f_rk x_rk >= d_k  (πλήρης κάλυψη ζήτησης)
          Σ_k x_rk <= v_r ,  x_rk ακέραια >= 0

(β)  max  Σ_k g_k y_k - Σ_k Σ_r c_rk x_rk
     s.t. 0.95 d_k <= Σ_r f_rk x_rk <= 1.05 d_k
          y_k <= d_k ,  y_k <= Σ_r f_rk x_rk
          Σ_k x_rk <= v_r ,  x_rk ακέραια >= 0 , y_k >= 0

Χρησιμοποιείται η PuLP με τον επιλυτή CBC, όπως στις διαλέξεις.
"""
import pulp

# ------------------------------------------------ στιγμιότυπο δοκιμής
K, R = 3, 2                                   # 3 διαδρομές, 2 τύποι οχημάτων
ROUTES = range(1, K + 1)
TYPES = range(1, R + 1)

d = {1: 500, 2: 300, 3: 420}                  # ζήτηση ανά διαδρομή
g = {1: 25.0, 2: 30.0, 3: 20.0}               # έσοδο ανά επιβάτη
f = {(1, 1): 120, (1, 2): 100, (1, 3): 110,   # χωρητικότητα τύπου r στη k
     (2, 1):  80, (2, 2):  90, (2, 3):  75}
c = {(1, 1): 900., (1, 2): 850., (1, 3): 880.,  # κόστος τύπου r στη k
     (2, 1): 600., (2, 2): 640., (2, 3): 610.}
v = {1: 6, 2: 8}                              # διαθέσιμα οχήματα ανά τύπο

solver = pulp.PULP_CBC_CMD(msg=False)


# =====================================================================
# Μοντέλο (α): η ζήτηση εξυπηρετείται στο ακέραιο
# =====================================================================
def model_a():
    m = pulp.LpProblem('MMM_plhrhs_kalypsis', pulp.LpMaximize)

    # Μεταβλητές απόφασης: ακέραιος αριθμός οχημάτων τύπου r στη διαδρομή k
    x = pulp.LpVariable.dicts('x', (TYPES, ROUTES), lowBound=0, cat='Integer')

    # Τα έσοδα Σ g_k d_k είναι ΣΤΑΘΕΡΑ (όλη η ζήτηση εξυπηρετείται), οπότε η
    # μεγιστοποίηση του κέρδους ισοδυναμεί με ελαχιστοποίηση του κόστους.
    esoda = sum(g[k] * d[k] for k in ROUTES)
    m += esoda - pulp.lpSum(c[r, k] * x[r][k] for r in TYPES for k in ROUTES)

    for k in ROUTES:                          # πλήρης κάλυψη της ζήτησης
        m += pulp.lpSum(f[r, k] * x[r][k] for r in TYPES) >= d[k], f'zhthsh_{k}'
    for r in TYPES:                           # διαθέσιμος στόλος
        m += pulp.lpSum(x[r][k] for k in ROUTES) <= v[r], f'stolos_{r}'

    m.solve(solver)
    return m, x, esoda


# =====================================================================
# Μοντέλο (β): χωρητικότητα στο [95%, 105%] της ζήτησης
# =====================================================================
def model_b():
    m = pulp.LpProblem('MMM_euelikth_kalypsi', pulp.LpMaximize)

    x = pulp.LpVariable.dicts('x', (TYPES, ROUTES), lowBound=0, cat='Integer')
    y = pulp.LpVariable.dicts('y', ROUTES, lowBound=0)   # επιβάτες που εξυπηρετούνται

    m += (pulp.lpSum(g[k] * y[k] for k in ROUTES)
          - pulp.lpSum(c[r, k] * x[r][k] for r in TYPES for k in ROUTES))

    for k in ROUTES:
        cap = pulp.lpSum(f[r, k] * x[r][k] for r in TYPES)
        m += cap >= 0.95 * d[k], f'kato_orio_{k}'     # >= 95% της ζήτησης
        m += cap <= 1.05 * d[k], f'anw_orio_{k}'      # <= 105% της ζήτησης
        m += y[k] <= d[k], f'y_leq_d_{k}'             # όχι πάνω από τη ζήτηση
        m += y[k] <= cap, f'y_leq_cap_{k}'            # όχι πάνω από τις θέσεις
    for r in TYPES:
        m += pulp.lpSum(x[r][k] for k in ROUTES) <= v[r], f'stolos_{r}'

    m.solve(solver)
    return m, x, y


# =====================================================================
ma, xa, esoda = model_a()
print('=== Μοντέλο (α): πλήρης κάλυψη της ζήτησης ===')
print(f' κατάσταση: {pulp.LpStatus[ma.status]}')
print('           ' + ''.join(f'{"διαδρ."+str(k):>12}' for k in ROUTES))
for r in TYPES:
    print(f' τύπος {r}:  ' + ''.join(f'{int(xa[r][k].value()):>12}' for k in ROUTES))
cap_a = {k: sum(f[r, k] * xa[r][k].value() for r in TYPES) for k in ROUTES}
print(' χωρητικότ.' + ''.join(f'{int(cap_a[k]):>12}' for k in ROUTES))
print(' ζήτηση   ' + ''.join(f'{d[k]:>12}' for k in ROUTES)
      + f'   (cap >= d παντού: {all(cap_a[k] >= d[k] for k in ROUTES)})')
kostos = esoda - pulp.value(ma.objective)
print(f' έσοδα = {esoda:.2f},  κόστος = {kostos:.2f},'
      f'  ΚΕΡΔΟΣ = {pulp.value(ma.objective):.2f}')

mb, xb, yb = model_b()
print('\n=== Μοντέλο (β): χωρητικότητα στο [95%, 105%] της ζήτησης ===')
print(f' κατάσταση: {pulp.LpStatus[mb.status]}')
print('           ' + ''.join(f'{"διαδρ."+str(k):>12}' for k in ROUTES))
for r in TYPES:
    print(f' τύπος {r}:  ' + ''.join(f'{int(xb[r][k].value()):>12}' for k in ROUTES))
cap_b = {k: sum(f[r, k] * xb[r][k].value() for r in TYPES) for k in ROUTES}
print(' χωρητικότ.' + ''.join(f'{int(cap_b[k]):>12}' for k in ROUTES))
print(' επιτρεπτό ' + ''.join(f'{f"[{0.95*d[k]:.0f},{1.05*d[k]:.0f}]":>12}' for k in ROUTES))
print(' y_k       ' + ''.join(f'{int(yb[k].value()):>12}' for k in ROUTES))
print(' min(d,cap)' + ''.join(f'{int(min(d[k], cap_b[k])):>12}' for k in ROUTES)
      + '   <- ταυτίζονται, όπως αναμενόταν')
print(f' ΚΕΡΔΟΣ = {pulp.value(mb.objective):.2f}')

print('\nΠαρατήρηση: στο (β) το y_k δένεται αυτόματα στο min(d_k, χωρητικότητα)')
print('επειδή g_k > 0 και η αντικειμενική συνάρτηση το μεγιστοποιεί.')
