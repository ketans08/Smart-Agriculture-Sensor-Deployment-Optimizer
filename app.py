# app.py
# ---------------------------------------------------------------------
# Sensor Deployment Optimization Dashboard
# Robust metaheuristic comparison w/ DMPA‑GWO hybrid focus
# Innovations:
#   * Auto‑Suggest Relay / Spare Sensor Placement (connectivity repair)
#   * Interactive Failure Scenario Explorer (DMPA‑GWO)
#   * Multi‑Ring Sensor Halo: sensing & comm ranges
# Session-state persistence so that moving widgets (e.g., scenario slider)
# does NOT force a re-run of the heavy optimization unless parameters change.
# ---------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
import networkx as nx

# ---------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------
st.sidebar.title("Optimization Parameters")

uploaded      = st.sidebar.file_uploader("Environment CSV (type,x,y[,radius])", type="csv")
area_size     = st.sidebar.number_input("Area size (square side)", min_value=1, value=100, step=5)
sensor_count  = st.sidebar.number_input("Sensor count", min_value=1, value=30, step=1)
sensor_range  = st.sidebar.number_input("Sensor range", min_value=1, value=15, step=1)
comm_factor   = st.sidebar.slider("Comm range (× sensor range)", 1.0, 3.0, 1.5, 0.1)
comm_range    = comm_factor * sensor_range
obs_radius    = st.sidebar.number_input("Default obstacle radius", min_value=1, value=8, step=1)

st.sidebar.markdown("---")
iterations    = st.sidebar.number_input("Iterations", min_value=1, value=50, step=5)
pop_size      = st.sidebar.number_input("Population size", min_value=2, value=30, step=2)
alpha_weight  = st.sidebar.slider("Alpha weight (Static vs Robust blend)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### Monte Carlo Robustness")
mc_train      = st.sidebar.number_input("MC Samples (Training)",  min_value=1, value=20,  step=5)
mc_report     = st.sidebar.number_input("MC Samples (Reporting)", min_value=1, value=200, step=10)
failure_prob  = st.sidebar.slider("Sensor Failure Probability", 0.0, 1.0, 0.10, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### Relay / Explorer Options")
relay_budget      = st.sidebar.number_input("Relay Budget (auto‑suggest)", min_value=0, value=5, step=1)
fail_scenarios    = st.sidebar.number_input("Failure Scenarios (Explorer)", min_value=1, value=25, step=1)
auto_apply_relays = st.sidebar.checkbox("Apply Relays in Summary Metrics (DMPA only)", value=True)

st.sidebar.markdown("---")
rng_seed = st.sidebar.number_input("Random Seed", min_value=0, value=1234, step=1)

run_button = st.sidebar.button("Run Optimization")  # triggers recompute

# ---------------------------------------------------------------------
# Derived Globals
# ---------------------------------------------------------------------
dim         = 2 * sensor_count
bounds_low  = np.zeros(dim)
bounds_high = np.tile([area_size, area_size], sensor_count)
area_center = np.array([area_size/2, area_size/2])

# Master RNG (all reproducibility flows from this)
master_rng = np.random.default_rng(rng_seed)
DEFAULT_ENV_PATH='enviroment.csv'

# ---------------------------------------------------------------------
# Environment loader (cached on inputs)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_environment(uploaded_file, area_size, obs_radius, seed, default_path: str = DEFAULT_ENV_PATH):
    """
    Load targets & obstacles with simple 3-level priority:
        1. user-uploaded file
        2. local default CSV (default_path)
        3. synthetic random fallback (seeded)

    Returns
    -------
    targets   : (Nt,2) float ndarray
    obstacles : (No,3) float ndarray  [x, y, radius]
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # 1) Try uploaded file
    # --------------------------------------------------
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            df = None  # fall through to default

    # --------------------------------------------------
    # 2) Try default file on disk
    # --------------------------------------------------
    if df is None and os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path)
        except Exception:
            df = None  # fall through to random

    # --------------------------------------------------
    # 3) Synthetic fallback if no usable DataFrame
    # --------------------------------------------------
    if df is None:
        targets = rng.uniform(0, area_size, (50, 2))
        obstacles = np.hstack([
            rng.uniform(0, area_size, (8, 2)),
            np.full((8, 1), float(obs_radius), dtype=float)
        ])
        return targets.astype(float), obstacles.astype(float)

    # --------------------------------------------------
    # Parse DataFrame (case-insensitive columns)
    # Expected: type,x,y[,radius]
    # --------------------------------------------------
    colmap = {c.lower(): c for c in df.columns}
    tcol = colmap.get('type')
    xcol = colmap.get('x')
    ycol = colmap.get('y')
    rcol = colmap.get('radius')  # optional

    # If required columns missing → synthetic fallback
    if tcol is None or xcol is None or ycol is None:
        targets = rng.uniform(0, area_size, (50, 2))
        obstacles = np.hstack([
            rng.uniform(0, area_size, (8, 2)),
            np.full((8, 1), float(obs_radius), dtype=float)
        ])
        return targets.astype(float), obstacles.astype(float)

    # Normalize the type column to lowercase so 'Target', 'TARGET', etc. all work.
    typ = df[tcol].astype(str).str.lower()

    # Targets
    tmask = typ == 'target'
    targets = df.loc[tmask, [xcol, ycol]].to_numpy(dtype=float)

    # Obstacles
    omask = typ == 'obstacle'
    if omask.any():
        if rcol is not None:
            obstacles = df.loc[omask, [xcol, ycol, rcol]].to_numpy(dtype=float)
        else:
            xy = df.loc[omask, [xcol, ycol]].to_numpy(dtype=float)
            obstacles = np.hstack([
                xy,
                np.full((xy.shape[0], 1), float(obs_radius), dtype=float)
            ])
    else:
        obstacles = np.empty((0, 3), dtype=float)

    return targets, obstacles

targets, obstacles = load_environment(uploaded, area_size, obs_radius, rng_seed)

# ---------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------
def static_metrics(sensors: np.ndarray):
    """
    Compute static metrics on a given set of sensors (k,2).
    Returns: (coverage, over_cov, connectivity, overlap, boundary, static_fitness)
    """
    n = sensors.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0, 0, 0, 1.0  # degenerate penalty

    dt        = cdist(targets, sensors)
    cov_mask  = dt <= sensor_range
    coverage  = np.mean(np.any(cov_mask, axis=1))
    over_cov  = np.mean(np.sum(cov_mask, axis=1) > 1)

    pairwise  = cdist(sensors, sensors)
    overlap   = int(np.sum((pairwise < sensor_range/2) & (pairwise > 0)) // 2)

    boundary  = int(np.sum((sensors < 0) | (sensors > area_size)))

    base_d    = np.linalg.norm(sensors - area_center, axis=1)
    adj       = pairwise <= sensor_range   # static comm = sensor_range
    start     = set(np.where(base_d <= sensor_range)[0])
    visited   = set(start)
    queue     = list(start)
    while queue:
        u = queue.pop(0)
        nbrs = np.where(adj[u])[0]
        for v in nbrs:
            if v not in visited:
                visited.add(v); queue.append(v)
    connectivity = len(visited) / n

    static_fitness = (
        -1.5 * coverage +
         0.01 * overlap +
         0.01 * boundary +
         0.5  * (1 - connectivity) +
         0.5  * over_cov
    )
    return coverage, over_cov, connectivity, overlap, boundary, static_fitness


def robust_penalty(agent: np.ndarray, rng: np.random.Generator, samples: int):
    """
    Monte Carlo penalty used *during optimization* (smaller sample count for speed).
    penal = (1-coverage) + (1-connectivity) over comm_range graph after random failures.
    """
    sensors  = agent.reshape(sensor_count, 2)
    pw       = cdist(sensors, sensors)
    adj_full = pw <= comm_range

    penalties = []
    for _ in range(samples):
        alive_mask = rng.random(sensor_count) > failure_prob
        if not np.any(alive_mask):
            penalties.append(1.0)
            continue
        surv = sensors[alive_mask]
        cov = np.mean(np.any(cdist(targets, surv) <= sensor_range, axis=1))

        idx = np.where(alive_mask)[0]
        base_mask = np.linalg.norm(sensors[idx] - area_center, axis=1) <= sensor_range
        start = set(idx[base_mask])
        visited = set(start)
        q = list(start)
        while q:
            u = q.pop(0)
            for v in idx:
                if v not in visited and adj_full[u, v]:
                    visited.add(v); q.append(v)
        conn = len(visited) / len(idx)
        penalties.append((1 - cov) + (1 - conn))

    return float(np.mean(penalties))


def monte_carlo_report(agent: np.ndarray, rng: np.random.Generator, samples: int):
    """
    Larger Monte Carlo evaluation for reporting.
    Returns dict: fit_mean/std, cov_mean/std, conn_mean/std, plus raw arrays.
    Fitness = static_fitness of survivors each trial (so lower is better).
    """
    sensors  = agent.reshape(sensor_count, 2)
    fits, covs, conns = [], [], []
    for _ in range(samples):
        alive_mask = rng.random(sensor_count) > failure_prob
        surv = sensors[alive_mask]
        cov, ov, conn, _, _, fit = static_metrics(surv)
        fits.append(fit); covs.append(cov); conns.append(conn)
    fits  = np.array(fits, dtype=float)
    covs  = np.array(covs, dtype=float)
    conns = np.array(conns, dtype=float)
    return {
        "fit_samples":  fits,
        "cov_samples":  covs,
        "conn_samples": conns,
        "fit_mean":     float(fits.mean())  if fits.size else np.nan,
        "fit_std":      float(fits.std(ddof=1)) if fits.size>1 else 0.0,
        "cov_mean":     float(covs.mean())  if covs.size else np.nan,
        "cov_std":      float(covs.std(ddof=1)) if covs.size>1 else 0.0,
        "conn_mean":    float(conns.mean()) if conns.size else np.nan,
        "conn_std":     float(conns.std(ddof=1)) if conns.size>1 else 0.0,
    }

# ---------------------------------------------------------------------
# Failure mask sampling
# ---------------------------------------------------------------------
def sample_failures(rng: np.random.Generator, n: int):
    return rng.random(n) > failure_prob

# ---------------------------------------------------------------------
# Build comm graph
# ---------------------------------------------------------------------
def build_comm_graph(sensors: np.ndarray, comm_range: float):
    n = sensors.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n <= 1: 
        return G
    pw = cdist(sensors, sensors)
    for i in range(n):
        for j in range(i+1, n):
            if pw[i,j] <= comm_range:
                G.add_edge(i,j, weight=float(pw[i,j]))
    return G

# ---------------------------------------------------------------------
# Auto‑Suggest Relay / Spare Sensor Placement
# ---------------------------------------------------------------------
def suggest_relays(sensors: np.ndarray,
                   alive_mask: np.ndarray,
                   comm_range: float,
                   budget: int):
    """
    Greedy relay placer (bug‑fixed):
    Connect disconnected *survivor* components by inserting up to `budget` relay
    nodes along the shortest inter‑component gap. Returns:

        relays (m,2)   array of relay coordinates  (m <= budget)
        G_final        nx.Graph built over survivors + relays

    Notes:
    * We only consider *alive* sensors as endpoints to connect.
    * After each batch of newly added relays we rebuild the working point
      array (`pts_current`) and the comm graph using survivors + all relays
      placed so far, so node IDs stay aligned with `pts_current`.
    * Distances between components are computed using `pts_current`, not the
      original survivor array (this was the source of the IndexError).
    """
    surv_pts = sensors[alive_mask]
    n_surv   = surv_pts.shape[0]

    # trivial cases
    if n_surv <= 1 or budget <= 0:
        return np.empty((0, 2), dtype=float), build_comm_graph(surv_pts, comm_range)

    relays = []                         # list of 2D points we place
    pts_current = surv_pts.copy()       # survivors + relays (grows each loop)
    budget_left = budget

    while budget_left > 0:
        # Build graph on current point set
        G = build_comm_graph(pts_current, comm_range)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(comps) <= 1:
            break  # already connected

        main = comps[0]         # largest component
        others = comps[1:]

        # Find nearest pair between main and any other component
        best_pair = None
        best_dist = np.inf
        for comp in others:
            main_idx = np.fromiter(main, dtype=int)
            comp_idx = np.fromiter(comp, dtype=int)
            dmat = cdist(pts_current[main_idx], pts_current[comp_idx])
            k = dmat.argmin()
            d = dmat.flat[k]
            if d < best_dist:
                best_dist = d
                i_main = main_idx[k // dmat.shape[1]]
                j_sub  = comp_idx[k %  dmat.shape[1]]
                best_pair = (i_main, j_sub)

        # Defensive: shouldn't happen, but break if no pair found
        if best_pair is None:
            break

        i_main, j_sub = best_pair
        p1 = pts_current[i_main]
        p2 = pts_current[j_sub]

        # How many relays needed to hop at most comm_range?
        needed = max(0, int(np.ceil(best_dist / comm_range)) - 1)

        if needed == 0:
            # Shouldn't occur because then they would've been connected,
            # but just in case: add an edge by rebuilding with a tiny jitter relay
            needed = 1

        use = min(needed, budget_left)
        for k in range(1, use + 1):
            t = k / (needed + 1)
            pos = p1 + t * (p2 - p1)
            relays.append(pos)

        # update current point set & budget
        pts_current = np.vstack([pts_current, np.vstack(relays[-use:])])
        budget_left -= use

        # loop continues: we’ll rebuild G at top of loop using pts_current

    G_final = build_comm_graph(pts_current, comm_range)
    return np.array(relays, dtype=float), G_final


# ---------------------------------------------------------------------
# Algorithm Implementations
# Each returns: best_agent, convergence_list
# ---------------------------------------------------------------------
def run_mo_pso_dmpa_gwo_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, dim))
    vel = np.zeros_like(pop)

    def obj(a):
        return (alpha_weight * static_metrics(a.reshape(sensor_count,2))[5] +
                (1-alpha_weight) * robust_penalty(a, rng, mc_train))

    pbest        = pop.copy()
    pbest_scores = np.array([obj(a) for a in pop])
    gbest_idx    = np.argmin(pbest_scores)

    pack_count = 5
    pack_size  = max(1, pop_size // pack_count)
    pack_idxs  = [np.arange(i*pack_size, min((i+1)*pack_size, pop_size)) for i in range(pack_count)]

    converg = []
    best_so_far = np.inf
    for t in range(iterations+1):
        gbest = pbest[gbest_idx]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(sensor_count,2)[alive_mask]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far:
            best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations:
            break

        a  = 2 - 2*t/iterations
        wg = 0.5 - 0.4*t/iterations

        # pack leaders (top3 per pack)
        leaders = []
        for idx in pack_idxs:
            pack_scores = pbest_scores[idx]
            top = idx[np.argsort(pack_scores)[:3]]
            leaders.append(pbest[top])

        for i in range(pop_size):
            r1, r2 = rng.random(dim), rng.random(dim)
            vel[i] = (0.4*vel[i] 
                      + 1.5*r1*(pbest[i] - pop[i]) 
                      + 1.5*r2*(pbest[gbest_idx] - pop[i]))
            # find pack
            pack_id = next(p for p, idx in enumerate(pack_idxs) if i in idx)
            alpha, beta, delta = leaders[pack_id]
            A = 2*a*rng.random(dim) - a
            C = 2*rng.random(dim)
            Xg = (alpha - A*np.abs(C*alpha - pop[i]) +
                  beta  - A*np.abs(C*beta  - pop[i]) +
                  delta - A*np.abs(C*delta - pop[i])) / 3.0
            vel[i] += wg*(Xg - pop[i])
            pop[i]  = np.clip(pop[i] + vel[i], bounds_low, bounds_high)

        scores = np.array([obj(a) for a in pop])
        improved = scores < pbest_scores
        pbest[improved]        = pop[improved]
        pbest_scores[improved] = scores[improved]
        gbest_idx = np.argmin(pbest_scores)

    return pbest[gbest_idx], converg


def run_robust_pso_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, dim))
    vel = np.zeros_like(pop)
    pbest        = pop.copy()
    pbest_scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
    gbest_idx    = np.argmin(pbest_scores)

    converg = []
    best_so_far = np.inf
    for t in range(iterations+1):
        gbest = pbest[gbest_idx]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(sensor_count,2)[alive_mask]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break

        for i in range(pop_size):
            r1, r2 = rng.random(), rng.random()
            vel[i] = (0.7*vel[i]
                      + 1.5*r1*(pbest[i] - pop[i])
                      + 1.5*r2*(pbest[gbest_idx] - pop[i]))
            pop[i]  = np.clip(pop[i] + vel[i], bounds_low, bounds_high)
            sc = robust_penalty(pop[i], rng, mc_train)
            if sc < pbest_scores[i]:
                pbest_scores[i] = sc
                pbest[i] = pop[i].copy()
        gbest_idx = np.argmin(pbest_scores)

    return pbest[gbest_idx], converg


def run_robust_gwo_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, dim))
    converg = []
    best_so_far = np.inf
    for t in range(iterations+1):
        scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
        gbest   = pop[np.argmin(scores)]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(sensor_count,2)[alive_mask]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break

        idx = np.argsort(scores)
        alpha, beta, delta = pop[idx[:3]]
        a = 2 - 2*t/iterations
        new_pop = []
        for agent in pop:
            A = 2*a*master_rng.random(dim) - a
            C = 2*master_rng.random(dim)
            X1 = alpha - A*np.abs(C*alpha - agent)
            X2 = beta  - A*np.abs(C*beta  - agent)
            X3 = delta - A*np.abs(C*delta - agent)
            new_pop.append((X1+X2+X3)/3.0)
        pop = np.clip(np.array(new_pop), bounds_low, bounds_high)

    scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
    return pop[np.argmin(scores)], converg


def run_robust_de_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, dim))
    scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
    converg = []
    best_so_far = np.inf
    for t in range(iterations+1):
        gbest = pop[np.argmin(scores)]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(sensor_count,2)[alive_mask]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break

        F, Cr = 0.8, 0.9
        for i in range(pop_size):
            idxs = rng.choice(pop_size, 3, replace=False)
            a, b, c = pop[idxs]
            donor = np.clip(a + F*(b - c), bounds_low, bounds_high)
            mask  = rng.random(dim) < Cr
            trial = np.where(mask, donor, pop[i])
            sc = robust_penalty(trial, rng, mc_train)
            if sc < scores[i]:
                pop[i] = trial; scores[i] = sc

    return pop[np.argmin(scores)], converg


def run_robust_ga_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, dim))
    converg = []
    best_so_far = np.inf
    for t in range(iterations+1):
        scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
        gbest = pop[np.argmin(scores)]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(sensor_count,2)[alive_mask]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break

        new = []
        while len(new) < pop_size:
            i1, i2 = rng.choice(pop_size, 2, replace=False)
            a, b = pop[i1], pop[i2]
            pt = rng.integers(1, dim)
            o1 = np.concatenate([a[:pt], b[pt:]])
            o2 = np.concatenate([b[:pt], a[pt:]])
            mut_mask = rng.random(dim) < 0.01
            o1[mut_mask] = rng.uniform(bounds_low[mut_mask], bounds_high[mut_mask])
            mut_mask = rng.random(dim) < 0.01
            o2[mut_mask] = rng.uniform(bounds_low[mut_mask], bounds_high[mut_mask])
            new.extend([o1, o2])
        pop = np.array(new[:pop_size])

    scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
    return pop[np.argmin(scores)], converg


def run_robust_woa_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, dim))
    best = pop[0].copy()
    best_score = robust_penalty(best, rng, mc_train)
    converg = []
    best_so_far = np.inf
    for t in range(iterations+1):
        alive_mask = sample_failures(rng, sensor_count)
        surv = best.reshape(sensor_count,2)[alive_mask]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break

        a = 2 - 2*t/iterations
        for i in range(pop_size):
            sc = robust_penalty(pop[i], rng, mc_train)
            if sc < best_score:
                best_score = sc
                best = pop[i].copy()
        for i in range(pop_size):
            r = rng.random()
            A = 2*a*rng.random(dim) - a
            C = 2*rng.random(dim)
            if r < 0.5:
                pop[i] = best - A*np.abs(C*best - pop[i])
            else:
                l = rng.uniform(-1,1,dim)
                pop[i] = np.abs(best - pop[i]) * np.exp(l) * np.cos(2*np.pi*l) + best
        pop = np.clip(pop, bounds_low, bounds_high)

    return best, converg


# ---------------------------------------------------------------------
# Runner dict
# ---------------------------------------------------------------------
runners = {
    'DMPA‑GWO': run_mo_pso_dmpa_gwo_with_convergence,
    'PSO':      run_robust_pso_with_convergence,
    'GWO':      run_robust_gwo_with_convergence,
    'DE':       run_robust_de_with_convergence,
    'GA':       run_robust_ga_with_convergence,
    'WOA':      run_robust_woa_with_convergence,
}

# ---------------------------------------------------------------------
# Multi‑Ring Before/After Plot
# ---------------------------------------------------------------------
def plot_before_after(name, sensors, alive_mask, comm_range, obstacles, targets,
                      sensor_range, area_size):
    """
    Return a Matplotlib figure (1x2) showing before vs after failure.
    Multi‑ring halo: inner = sensing, outer ring = comm range.
    """
    n = sensors.shape[0]
    G_full = build_comm_graph(sensors, comm_range)
    mst_full = nx.minimum_spanning_tree(G_full)

    surv_idx = np.where(alive_mask)[0]
    G_surv = G_full.subgraph(surv_idx)
    mst_surv = nx.minimum_spanning_tree(G_surv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle(f"{name}: Before vs After Failure", fontsize=16, weight='bold')

    def _draw(ax, title, mst_edges, show_failed=False):
        ax.set_title(title)
        # Obstacles
        for x,y,r in obstacles:
            ax.add_patch(Circle((x,y), r, edgecolor='red', facecolor='red', alpha=0.3))
        # Coverage & comm halos
        for x,y in sensors:
            # sensing
            ax.add_patch(Circle((x,y), sensor_range,
                                edgecolor='purple', facecolor='purple', alpha=0.15, linewidth=0.5))
            # comm ring only outline
            ax.add_patch(Circle((x,y), comm_range,
                                edgecolor='cyan', facecolor='none', alpha=0.5, linestyle='--', linewidth=0.8))
        # MST edges
        edge_color = 'blue' if show_failed else 'gray'
        for u,v in mst_edges.edges():
            ax.plot([sensors[u,0], sensors[v,0]],
                    [sensors[u,1], sensors[v,1]],
                    color=edge_color, linewidth=2)
        # Sensors
        if show_failed:
            surv = sensors[alive_mask]
            ax.scatter(surv[:,0], surv[:,1],
                       s=80, edgecolor='white', facecolor='none', linewidth=1.5, label='Sensors')
            fail = sensors[~alive_mask]
            if fail.size > 0:
                ax.scatter(fail[:,0], fail[:,1],
                           marker='x', c='red', s=100, linewidths=2, label='Failed')
        else:
            ax.scatter(sensors[:,0], sensors[:,1],
                       s=80, edgecolor='white', facecolor='none', linewidth=1.5, label='Sensors')
        # Targets
        ax.scatter(targets[:,0], targets[:,1], c='green', s=40, label='Targets')
        ax.set_xlim(0, area_size); ax.set_ylim(0, area_size)
        ax.set_aspect('equal'); ax.grid(True, linestyle='--', alpha=0.3)

    _draw(ax1, "Before Failure", mst_full, show_failed=False)
    _draw(ax2, "After Failure",  mst_surv, show_failed=True)

    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Targets'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='none', markeredgecolor='purple', markersize=8, label='Sensors'),
        Line2D([0],[0], marker='x', color='red', markersize=10, label='Failed'),
        Patch(facecolor='red', edgecolor='red', alpha=0.3, label='Obstacles'),
        Patch(facecolor='purple', edgecolor='purple', alpha=0.15, label='Sense Range'),
        Line2D([0],[0], color='cyan', linestyle='--', label='Comm Range'),
    ]
    ax2.legend(handles=handles, loc='upper right')
    fig.tight_layout(rect=[0,0.03,1,0.95])
    return fig

# ---------------------------------------------------------------------
# Evaluate all algorithms (single run each)
# ---------------------------------------------------------------------
def evaluate_all_algorithms(rng: np.random.Generator):
    results = {}
    child_seeds = rng.integers(0, 2**32-1, size=len(runners))
    for (name, fn), seed in zip(runners.items(), child_seeds):
        sub_rng = np.random.default_rng(seed)
        best_agent, converg = fn(sub_rng)
        sensors = best_agent.reshape(sensor_count,2)

        # sample a failure scenario (for baseline plots & sample metrics)
        alive_mask = sample_failures(sub_rng, sensor_count)
        surv = sensors[alive_mask]

        metrics_before = static_metrics(sensors)
        metrics_after_sample = static_metrics(surv)

        # MC reporting
        mc_rng = np.random.default_rng(seed + 999)
        metrics_mc = monte_carlo_report(best_agent, mc_rng, mc_report)

        results[name] = {
            'best_agent':           best_agent,
            'convergence':          converg,
            'sample_alive_mask':    alive_mask,
            'metrics_before':       metrics_before,
            'metrics_after_sample': metrics_after_sample,
            'metrics_mc':           metrics_mc,
        }
    return results

# ---------------------------------------------------------------------
# Summary Table Formatting
# ---------------------------------------------------------------------
ARROW = "→"
def fmt_ba(b, a, prec=2, signed=False):
    fmt = f"{{:+.{prec}f}}" if signed else f"{{:.{prec}f}}"
    return f"{fmt.format(b)} {ARROW} {fmt.format(a)}"

# ---------------------------------------------------------------------
# Interactive Failure Scenario Explorer (DMPA‑GWO only)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def precompute_failure_scenarios(n_scenarios, seed, n_sensors, failure_prob):
    rng = np.random.default_rng(seed)
    return (rng.random((n_scenarios, n_sensors)) > failure_prob).astype(bool)

def explorer_plot(sensors, alive_mask, relays, comm_range, obstacles, targets, sensor_range, area_size):
    """
    Plot a *single* panel: survivors + relays + failed sensors.
    Multi‑ring halos drawn for original sensors; relays get orange halos.
    """
    G_base = build_comm_graph(sensors[alive_mask], comm_range)

    # include relays in graph & layout metrics
    if relays.size > 0:
        all_pts = np.vstack([sensors[alive_mask], relays])
        G = build_comm_graph(all_pts, comm_range)
    else:
        G = G_base
        all_pts = sensors[alive_mask]

    mst = nx.minimum_spanning_tree(G)

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_title("Failure Scenario Explorer", fontsize=14, weight='bold')

    # Obstacles
    for x,y,r in obstacles:
        ax.add_patch(Circle((x,y), r, edgecolor='red', facecolor='red', alpha=0.3))

    # Original sensors halos
    for (x,y), alive in zip(sensors, alive_mask):
        ax.add_patch(Circle((x,y), sensor_range,
                            edgecolor='purple' if alive else 'gray',
                            facecolor='purple' if alive else 'none',
                            alpha=0.10 if alive else 0.0,
                            linewidth=0.5))
        ax.add_patch(Circle((x,y), comm_range,
                            edgecolor='cyan', facecolor='none',
                            alpha=0.5 if alive else 0.15,
                            linestyle='--', linewidth=0.8))

    # Relay halos
    for x,y in relays:
        ax.add_patch(Circle((x,y), sensor_range,
                            edgecolor='orange', facecolor='orange', alpha=0.15, linewidth=0.5))
        ax.add_patch(Circle((x,y), comm_range,
                            edgecolor='orange', facecolor='none', linestyle=':', alpha=0.6, linewidth=0.8))

    # MST edges (over survivors + relays)
    pts = all_pts
    for u,v in mst.edges():
        x0,y0 = pts[u]; x1,y1 = pts[v]
        ax.plot([x0,x1],[y0,y1],color='blue',linewidth=2)

    # Plot survivors/failed
    surv_pts = sensors[alive_mask]
    ax.scatter(surv_pts[:,0], surv_pts[:,1], s=80, edgecolor='white', facecolor='none', linewidth=1.5, label='Alive')
    fail_pts = sensors[~alive_mask]
    if fail_pts.size>0:
        ax.scatter(fail_pts[:,0], fail_pts[:,1], marker='x', c='red', s=100, linewidths=2, label='Failed')
    if relays.size>0:
        ax.scatter(relays[:,0], relays[:,1], marker='*', c='orange', s=140, edgecolor='black', linewidths=0.8, label='Relays')

    # Targets
    ax.scatter(targets[:,0], targets[:,1], c='green', s=40, label='Targets')
    ax.set_xlim(0, area_size); ax.set_ylim(0, area_size)
    ax.set_aspect('equal'); ax.grid(True, linestyle='--', alpha=0.3)

    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------
# Main Execution Flow w/ Session State Persistence
# ---------------------------------------------------------------------
st.title("📡 Sensor Deployment Optimization Dashboard")

# Build a parameter signature so we know when to invalidate cached results.
param_sig = (
    area_size, sensor_count, sensor_range, comm_factor, obs_radius,
    iterations, pop_size, alpha_weight, mc_train, mc_report,
    failure_prob, relay_budget, fail_scenarios, rng_seed
)

# If RUN pressed *or* results missing *or* params changed -> recompute.
if run_button or ('_param_sig' not in st.session_state) or (st.session_state['_param_sig'] != param_sig):
    with st.spinner("Running optimizations… (this can take a minute)"):
        results = evaluate_all_algorithms(master_rng)
    st.session_state['results']    = results
    st.session_state['_param_sig'] = param_sig
    # precompute failure masks for explorer & store
    st.session_state['fail_masks'] = precompute_failure_scenarios(
        fail_scenarios, rng_seed+2025, sensor_count, failure_prob
    )

# Pull from session_state (may be None if never run)
results = st.session_state.get('results', None)

if results is None:
    st.info("Configure parameters in the sidebar and click **Run Optimization**.")
    st.stop()

# explorer failure masks
fail_masks = st.session_state.get('fail_masks')
# If user changed just fail_scenarios slider (w/out run), regenerate quickly:
if fail_masks is None or fail_masks.shape[0] != fail_scenarios or fail_masks.shape[1] != sensor_count:
    fail_masks = precompute_failure_scenarios(fail_scenarios, rng_seed+2025, sensor_count, failure_prob)
    st.session_state['fail_masks'] = fail_masks

# ---------------- Convergence data ----------------
conv_df = pd.DataFrame({name: results[name]['convergence'] for name in runners})
conv_df.index.name = "Iteration"
bsf_df = conv_df.cummin()

st.subheader("Convergence Comparison (Best‑so‑Far After‑Failure Static Fitness)")
st.line_chart(bsf_df)

# ---------------- Before/After Plots (with multi‑ring halos) ----------------
st.subheader("Before vs After Failure Deployments")
for name in runners.keys():
    data = results[name]
    sensors = data['best_agent'].reshape(sensor_count,2)
    alive_mask = data['sample_alive_mask']
    fig = plot_before_after(name, sensors, alive_mask,
                            comm_range, obstacles, targets,
                            sensor_range, area_size)
    st.pyplot(fig)

# ---------------- DMPA‑GWO Relay Analysis (using its sampled failure) ----------------
st.subheader("DMPA‑GWO Relay Repair Analysis")

dm_data    = results['DMPA‑GWO']
dm_sensors = dm_data['best_agent'].reshape(sensor_count,2)
dm_alive   = dm_data['sample_alive_mask']
dm_surv    = dm_sensors[dm_alive]

cov_b, ov_b, conn_b, ovl_b, bnd_b, sf_b = dm_data['metrics_before']
cov_a, ov_a, conn_a, ovl_a, bnd_a, sf_a = dm_data['metrics_after_sample']

# Suggest relays for this failure
relays, G_rel = suggest_relays(dm_sensors, dm_alive, comm_range, relay_budget)
if relays.size > 0:
    repaired = np.vstack([dm_surv, relays])
    cov_r, ov_r, conn_r, ovl_r, bnd_r, sf_r = static_metrics(repaired)
else:
    cov_r, ov_r, conn_r, ovl_r, bnd_r, sf_r = cov_a, ov_a, conn_a, ovl_a, bnd_a, sf_a

cols = st.columns(3)
cols[0].metric("Coverage",   f"{cov_b:.2f} → {cov_a:.2f}", f"{cov_r - cov_a:+.2f} (relay)")
cols[1].metric("Conn",       f"{conn_b:.3f} → {conn_a:.3f}", f"{conn_r - conn_a:+.3f} (relay)")
cols[2].metric("Static Fit", f"{sf_b:+.3f} → {sf_a:+.3f}", f"{sf_r - sf_a:+.3f} (relay)")

fig_rel = explorer_plot(dm_sensors, dm_alive, relays,
                        comm_range, obstacles, targets,
                        sensor_range, area_size)
st.pyplot(fig_rel)

# ---------------- Interactive Failure Scenario Explorer (DMPA‑GWO) ----------------
st.subheader("🔍 Failure Scenario Explorer (DMPA‑GWO)")
scen_idx = st.slider("Scenario #", 0, fail_scenarios-1, 0, key="scenario_slider")
scen_alive = fail_masks[scen_idx]

scen_surv = dm_sensors[scen_alive]
cov_s, ov_s, conn_s, ovl_s, bnd_s, sf_s = static_metrics(scen_surv)
relays_s, _ = suggest_relays(dm_sensors, scen_alive, comm_range, relay_budget)
if relays_s.size > 0:
    scen_rep = np.vstack([scen_surv, relays_s])
    cov_sr, ov_sr, conn_sr, ovl_sr, bnd_sr, sf_sr = static_metrics(scen_rep)
else:
    cov_sr, ov_sr, conn_sr, ovl_sr, bnd_sr, sf_sr = cov_s, ov_s, conn_s, ovl_s, bnd_s, sf_s

exp_cols = st.columns(3)
exp_cols[0].metric("Coverage",   f"{cov_s:.2f}", f"{cov_sr - cov_s:+.2f} (relay)")
exp_cols[1].metric("Conn",       f"{conn_s:.3f}", f"{conn_sr - conn_s:+.3f} (relay)")
exp_cols[2].metric("Static Fit", f"{sf_s:+.3f}", f"{sf_sr - sf_s:+.3f} (relay)")

fig_expl = explorer_plot(dm_sensors, scen_alive, relays_s,
                         comm_range, obstacles, targets,
                         sensor_range, area_size)
st.pyplot(fig_expl)

# ---------------- Summary Table ----------------
st.subheader("📋 Summary Table")

final_bsf = bsf_df.iloc[-1]  # series: alg -> final best‑so‑far convergence

rows = []
for name, data in results.items():
    covB, ovB, connB, ovlB, bndB, sfB = data['metrics_before']
    covA, ovA, connA, ovlA, bndA, sfA = data['metrics_after_sample']
    mc = data['metrics_mc']

    # If DMPA and auto_apply_relays: use repaired metrics
    if name == 'DMPA‑GWO' and auto_apply_relays:
        covA, ovA, connA, ovlA, bndA, sfA = cov_r, ov_r, conn_r, ovl_r, bnd_r, sf_r

    rows.append({
        "Algorithm":                name,
        "StaticFit B→A":            fmt_ba(sfB, sfA, prec=3, signed=True),
        "ExpFit After (MC)":        mc['fit_mean'],
        "Fit Std (MC)":             mc['fit_std'],
        "Coverage B→A":             fmt_ba(covB, covA, prec=2),
        "ExpCov After (MC)":        mc['cov_mean'],
        "Cov Std (MC)":             mc['cov_std'],
        "Connectivity B→A":         fmt_ba(connB, connA, prec=3),
        "ExpConn After (MC)":       mc['conn_mean'],
        "Conn Std (MC)":            mc['conn_std'],
        "OverCov B→A":              fmt_ba(ovB, ovA, prec=2),
        "Overlap B→A":              f"{ovlB} {ARROW} {ovlA}",
        "Boundary B→A":             f"{bndB} {ARROW} {bndA}",
        "Final Conv (best‑so‑far)": final_bsf[name],
    })

summary_df = pd.DataFrame(rows).set_index("Algorithm")
mc_cols = [
    "ExpFit After (MC)", "Fit Std (MC)",
    "ExpCov After (MC)", "Cov Std (MC)",
    "ExpConn After (MC)", "Conn Std (MC)",
    "Final Conv (best‑so‑far)"
]
summary_df[mc_cols] = summary_df[mc_cols].apply(pd.to_numeric, errors='coerce')

st.dataframe(summary_df, use_container_width=True)
st.success("Done!")
