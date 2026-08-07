import math
import heapq
import numpy as np


# ============================================================
# 1. Cohésion de Gras
# ============================================================

def g_pair(p: float) -> float:
    """
    Cohésion d'ordre 2 utilisée dans les simulations.

    g(p) = sqrt(1 - H(p)/2), si p >= 0.5
         = 0, sinon,
    où H est l'entropie binaire en base 2.
    """
    p = float(p)

    if p < 0.5:
        return 0.0

    if p >= 1.0:
        H = 0.0
    else:
        H = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    return math.sqrt(max(0.0, 1.0 - H / 2.0))


def class_stats(C, M, cache=None):
    """
    Pour une classe ordonnée C, retourne :
        cohesion(C), p_min(C), p_max(C).

    Pour i < j, l'intensité interne utilisée est M[C[i], C[j]].
    """
    C = tuple(C)

    if cache is None:
        cache = {}

    if C in cache:
        return cache[C]

    if len(C) <= 1:
        ans = (1.0, 1.0, 1.0)
        cache[C] = ans
        return ans

    pvals = []
    gvals = []

    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            p = float(M[C[i], C[j]])
            pvals.append(p)
            gvals.append(g_pair(p))

    if any(v <= 0.0 for v in gvals):
        coh = 0.0
    else:
        coh = math.exp(sum(math.log(v) for v in gvals) / len(gvals))

    ans = (coh, min(pvals), max(pvals))
    cache[C] = ans
    return ans


def cohesion(C, M, cache=None):
    return class_stats(C, M, cache)[0]


def delta(C, M, cache=None):
    """
    Delta(C) = g(p_max(C)) - g(p_min(C)).
    """
    _, pmin, pmax = class_stats(C, M, cache)
    return g_pair(pmax) - g_pair(pmin)


# ============================================================
# 2. Matrice d'implication simulée
# ============================================================

def generate_implication_matrix(n, rng=None):
    """
    M_ij iid Uniform(0,1), diagonale nulle.
    """
    if rng is None:
        rng = np.random.default_rng()

    M = rng.random((n, n))
    np.fill_diagonal(M, 0.0)
    return M


# ============================================================
# 3. Algorithme de Gras - version expérimentale
# ============================================================

def gras_algorithm(M):
    """
    Implémentation expérimentale de Gras utilisée dans les simulations.

    Départ :
        classes actives = singletons.

    A chaque étape :
        toutes les concaténations ordonnées de deux classes actives
        sont considérées ;
        on retient celle de cohésion maximale.

    Arrêt :
        lorsque la meilleure cohésion est nulle.

    Retour :
        SG          : ensemble des classes composées sélectionnées ;
        selected    : liste ordonnée des classes sélectionnées ;
        active      : classes finales.
    """
    n = M.shape[0]

    active = [(i,) for i in range(n)]
    selected = []
    cache = {}

    while len(active) >= 2:

        best = None
        best_coh = 0.0

        for ia, A in enumerate(active):
            for ib, B in enumerate(active):

                if ia == ib:
                    continue

                C = A + B
                coh = cohesion(C, M, cache)

                if (
                    coh > best_coh + 1e-15
                    or (
                        abs(coh - best_coh) <= 1e-15
                        and best is not None
                        and C < best[2]
                    )
                ):
                    best = (ia, ib, C)
                    best_coh = coh

        if best is None or best_coh <= 0.0:
            break

        ia, ib, C = best

        selected.append(C)

        active = [
            chain
            for k, chain in enumerate(active)
            if k not in (ia, ib)
        ]

        active.append(C)

    return set(selected), selected, active


# ============================================================
# 4. Glouton par fusion - utilisé pour les comparaisons SG/SN
# ============================================================

def greedy_algorithm(M, alpha=0.7):
    """
    Version gloutonne par fusions utilisée pour les expériences
    Jaccard / inclusion.

    Poids marginal :
        w(A,B) = M[end(A), start(B)].

    Deux indices sont calculés :
        R_top2   : version littérale "deux meilleurs candidats";
        R_strong : meilleur candidat comparé à tous les concurrents.

    lbar est défini opérationnellement comme la longueur moyenne
    des chaînes finales contenant au moins un sommet affecté.
    """
    n = M.shape[0]

    active = [(i,) for i in range(n)]
    selected = []
    cache = {}

    R_top2 = 0
    R_strong = 0

    affected_vertices_top2 = set()
    affected_vertices_strong = set()

    risk_log = []

    while len(active) >= 2:

        candidates = []

        for ia, A in enumerate(active):
            for ib, B in enumerate(active):

                if ia == ib:
                    continue

                w = float(M[A[-1], B[0]])

                if w >= alpha:
                    C = A + B
                    candidates.append((w, ia, ib, C))

        if not candidates:
            break

        candidates.sort(key=lambda x: (-x[0], x[3]))

        w1, ia1, ib1, C1 = candidates[0]

        g1 = g_pair(w1)
        d1 = delta(C1, M, cache)
        coh1 = cohesion(C1, M, cache)

        # ----------------------------------------------------
        # R_top2 : seulement les deux premiers candidats
        # ----------------------------------------------------

        top2_violation = False

        if len(candidates) >= 2:

            w2, _, _, C2 = candidates[1]
            d2 = delta(C2, M, cache)

            gap = abs(g1 - g_pair(w2))

            top2_violation = gap <= d1 + d2

            if top2_violation:
                R_top2 += 1
                affected_vertices_top2.update(C1)
                affected_vertices_top2.update(C2)

        # ----------------------------------------------------
        # R_strong : meilleur candidat contre tous les autres
        # ----------------------------------------------------

        violating_competitors = []

        for wj, _, _, Cj in candidates[1:]:

            dj = delta(Cj, M, cache)

            if g1 - g_pair(wj) <= d1 + dj:
                violating_competitors.append(Cj)

        stop_mismatch_risk = coh1 <= 0.0

        strong_violation = stop_mismatch_risk or bool(violating_competitors)

        if strong_violation:

            R_strong += 1

            affected_vertices_strong.update(C1)

            for Cj in violating_competitors:
                affected_vertices_strong.update(Cj)

        risk_log.append(
            {
                "step": len(selected) + 1,
                "C1": C1,
                "w1": w1,
                "coh_C1": coh1,
                "top2_violation": top2_violation,
                "strong_violation": strong_violation,
                "stop_mismatch_risk": stop_mismatch_risk,
            }
        )

        # Choix glouton
        selected.append(C1)

        active = [
            chain
            for k, chain in enumerate(active)
            if k not in (ia1, ib1)
        ]

        active.append(C1)

    affected_lengths_top2 = [
        len(chain)
        for chain in active
        if set(chain) & affected_vertices_top2
    ]

    affected_lengths_strong = [
        len(chain)
        for chain in active
        if set(chain) & affected_vertices_strong
    ]

    lbar_top2 = (
        float(np.mean(affected_lengths_top2))
        if affected_lengths_top2
        else 0.0
    )

    lbar_strong = (
        float(np.mean(affected_lengths_strong))
        if affected_lengths_strong
        else 0.0
    )

    return (
        set(selected),
        selected,
        active,
        R_top2,
        lbar_top2,
        R_strong,
        lbar_strong,
        risk_log,
    )


# ============================================================
# 5. Glouton optimisé par tas
# ============================================================

def greedy_heap_full(M, alpha=0.7):
    """
    Version optimisée utilisée pour les benchmarks.

    Pour chaque paire non ordonnée {i,j} :
        on garde l'orientation de plus forte intensité.

    Les arêtes candidates sont placées dans un tas.
    Contraintes :
        deg+(x) <= 1,
        deg-(x) <= 1,
        pas de cycle.

    Retour :
        classes : tous les intervalles des chemins finaux,
                  de longueur >= 2 ;
        paths   : chemins finaux ;
        edges   : arêtes retenues.
    """
    n = M.shape[0]

    H = []

    for i in range(n):
        for j in range(i + 1, n):

            if M[i, j] >= M[j, i]:
                v, a, b = float(M[i, j]), i, j
            else:
                v, a, b = float(M[j, i]), j, i

            if v >= alpha:
                heapq.heappush(H, (-v, a, b))

    out_used = np.zeros(n, dtype=bool)
    in_used = np.zeros(n, dtype=bool)

    succ = [-1] * n
    pred = [-1] * n

    def creates_cycle(x, y):
        """
        Tester s'il existe déjà y -> ... -> x.
        """
        z = y

        while z != -1:

            if z == x:
                return True

            z = succ[z]

        return False

    edges = []

    while H:

        negv, x, y = heapq.heappop(H)

        if out_used[x] or in_used[y]:
            continue

        if creates_cycle(x, y):
            continue

        succ[x] = y
        pred[y] = x

        out_used[x] = True
        in_used[y] = True

        edges.append((x, y))

    starts = [
        i
        for i in range(n)
        if pred[i] == -1
    ]

    paths = []
    visited = np.zeros(n, dtype=bool)

    for s in starts:

        path = []
        z = s

        while z != -1 and not visited[z]:

            path.append(z)
            visited[z] = True
            z = succ[z]

        if path:
            paths.append(tuple(path))

    for i in range(n):

        if not visited[i]:
            paths.append((i,))

    classes = set()

    for path in paths:

        L = len(path)

        for i in range(L):
            for j in range(i + 2, L + 1):
                classes.add(path[i:j])

    return classes, paths, edges


# ============================================================
# 6. Distances et indices
# ============================================================

def jaccard_dissimilarity(SG, SN):
    """
    D_J(SG,SN) = 1 - |SG intersect SN| / |SG union SN|.
    """
    union = SG | SN

    if len(union) == 0:
        return 0.0

    return 1.0 - len(SG & SN) / len(union)


def directional_distances(SG, SN):
    """
    Mesures directionnelles :

    D_G = |SG \\ SN| / |SG|
        = 1 - |SG intersect SN| / |SG|

    D_N = |SN \\ SG| / |SN|
        = 1 - |SG intersect SN| / |SN|
    """
    inter = SG & SN

    if len(SG) == 0:
        D_G = 0.0
        recall_G = 1.0
    else:
        D_G = len(SG - SN) / len(SG)
        recall_G = len(inter) / len(SG)

    if len(SN) == 0:
        D_N = 0.0
        precision_N = 1.0
    else:
        D_N = len(SN - SG) / len(SN)
        precision_N = len(inter) / len(SN)

    return {
        "D_G": D_G,
        "D_N": D_N,
        "recall_G": recall_G,
        "precision_N": precision_N,
        "SG_subset_SN": SG.issubset(SN),
    }


# ============================================================
# 7. Nombre exact de classes à partir des chemins
# ============================================================

def class_count_from_paths(paths, include_singletons=True):
    """
    Si un chemin a L sommets :
        nombre d'intervalles = L(L+1)/2

    Si include_singletons=False :
        on soustrait L, donc seulement les classes de taille >= 2.
    """
    total = 0

    for path in paths:

        L = len(path)

        if include_singletons:
            total += L * (L + 1) / 2
        else:
            total += L * (L - 1) / 2

    return total
