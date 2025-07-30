# app.py
# ---------------------------------------------------------------------
# Sensor Deployment Optimization Dashboard
# Robust metaheuristic comparison w/ DMPA‑GWO hybrid focus
# ---------------------------------------------------------------------

import os
import json
import time
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
import networkx as nx

# -------------------- Persistence folder -----------------------------
PERSIST_DIR = Path(".persist")
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Control whether freshly recomputed results are saved to disk
SAVE_NEW_RUNS = False  # DO NOT save new runs; set True to re-enable
# ---------------------------------------------------------------------

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
DEFAULT_ENV_PATH = 'environment.csv'  # fixed spelling

# ---------------------------------------------------------------------
# Environment loader (cached on inputs)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_environment(uploaded_file, area_size, obs_radius, seed, default_path: str = DEFAULT_ENV_PATH):
    """
    Load targets & obstacles with 3-level priority:
        1. user-uploaded file
        2. local default CSV (default_path)
        3. synthetic random fallback (seeded)
    """
    rng = np.random.default_rng(seed)

    # 1) Uploaded file
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            df = None

    # 2) Default file
    if df is None and os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path)
        except Exception:
            df = None

    # 3) Synthetic fallback
    if df is None:
        targets = rng.uniform(0, area_size, (50, 2))
        obstacles = np.hstack([
            rng.uniform(0, area_size, (8, 2)),
            np.full((8, 1), float(obs_radius), dtype=float)
        ])
        return targets.astype(float), obstacles.astype(float)

    # Parse (type,x,y[,radius])
    colmap = {c.lower(): c for c in df.columns}
    tcol = colmap.get('type'); xcol = colmap.get('x'); ycol = colmap.get('y'); rcol = colmap.get('radius')
    if tcol is None or xcol is None or ycol is None:
        targets = rng.uniform(0, area_size, (50, 2))
        obstacles = np.hstack([
            rng.uniform(0, area_size, (8, 2)),
            np.full((8, 1), float(obs_radius), dtype=float)
        ])
        return targets.astype(float), obstacles.astype(float)

    typ = df[tcol].astype(str).str.lower()
    targets = df.loc[typ == 'target', [xcol, ycol]].to_numpy(dtype=float)

    omask = typ == 'obstacle'
    if omask.any():
        if rcol is not None:
            obstacles = df.loc[omask, [xcol, ycol, rcol]].to_numpy(dtype=float)
        else:
            xy = df.loc[omask, [xcol, ycol]].to_numpy(dtype=float)
            obstacles = np.hstack([xy, np.full((xy.shape[0], 1), float(obs_radius), dtype=float)])
    else:
        obstacles = np.empty((0, 3), dtype=float)
    return targets, obstacles

targets, obstacles = load_environment(uploaded, area_size, obs_radius, rng_seed)

# ---------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------
def static_metrics(sensors: np.ndarray):
    sensors = np.asarray(sensors).reshape(-1, 2)
    n = sensors.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0, 0, 0, 1.0
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
    static_fitness = (-1.5 * coverage + 0.01 * overlap + 0.01 * boundary +
                      0.5  * (1 - connectivity) + 0.5  * over_cov)
    return coverage, over_cov, connectivity, overlap, boundary, static_fitness


def robust_penalty(agent: np.ndarray, rng: np.random.Generator, samples: int):
    sensors  = agent.reshape(-1, 2)
    pw       = cdist(sensors, sensors)
    adj_full = pw <= comm_range

    penalties = []
    n = sensors.shape[0]
    for _ in range(samples):
        alive_mask = rng.random(n) > failure_prob
        if not np.any(alive_mask):
            penalties.append(1.0); continue
        surv = sensors[alive_mask]
        cov = np.mean(np.any(cdist(targets, surv) <= sensor_range, axis=1))
        idx = np.where(alive_mask)[0]
        base_mask = np.linalg.norm(sensors[idx] - area_center, axis=1) <= sensor_range
        start = set(idx[base_mask]); visited = set(start); q = list(start)
        while q:
            u = q.pop(0)
            for v in idx:
                if v not in visited and adj_full[u, v]:
                    visited.add(v); q.append(v)
        conn = len(visited) / len(idx)
        penalties.append((1 - cov) + (1 - conn))
    return float(np.mean(penalties))


def monte_carlo_report(agent: np.ndarray, rng: np.random.Generator, samples: int):
    sensors  = agent.reshape(-1, 2)
    n = sensors.shape[0]
    fits, covs, conns = [], [], []
    for _ in range(samples):
        alive_mask = rng.random(n) > failure_prob
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
    sensors = np.asarray(sensors).reshape(-1, 2)
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
def suggest_relays(sensors: np.ndarray, alive_mask: np.ndarray, comm_range: float, budget: int):
    sensors = np.asarray(sensors).reshape(-1, 2)
    alive_mask = np.asarray(alive_mask).astype(bool)
    alive_mask = alive_mask[:sensors.shape[0]]
    surv_pts = sensors[alive_mask]
    n_surv   = surv_pts.shape[0]
    if n_surv <= 1 or budget <= 0:
        return np.empty((0, 2), dtype=float), build_comm_graph(surv_pts, comm_range)
    relays = []; pts_current = surv_pts.copy(); budget_left = budget
    while budget_left > 0:
        G = build_comm_graph(pts_current, comm_range)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(comps) <= 1: break
        main = comps[0]; others = comps[1:]
        best_pair = None; best_dist = np.inf
        for comp in others:
            main_idx = np.fromiter(main, dtype=int)
            comp_idx = np.fromiter(comp, dtype=int)
            dmat = cdist(pts_current[main_idx], pts_current[comp_idx])
            k = dmat.argmin(); d = dmat.flat[k]
            if d < best_dist:
                best_dist = d
                i_main = main_idx[k // dmat.shape[1]]
                j_sub  = comp_idx[k %  dmat.shape[1]]
                best_pair = (i_main, j_sub)
        if best_pair is None: break
        i_main, j_sub = best_pair
        p1 = pts_current[i_main]; p2 = pts_current[j_sub]
        needed = max(0, int(np.ceil(best_dist / comm_range)) - 1)
        if needed == 0: needed = 1
        use = min(needed, budget_left)
        for k in range(1, use + 1):
            t = k / (needed + 1); pos = p1 + t * (p2 - p1); relays.append(pos)
        pts_current = np.vstack([pts_current, np.vstack(relays[-use:])])
        budget_left -= use
    G_final = build_comm_graph(pts_current, comm_range)
    return np.array(relays, dtype=float), G_final

# ---------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------
def run_mo_pso_dmpa_gwo_pp_with_convergence(rng: np.random.Generator):
    pack_count_init   = 5
    mutation_max      = 0.15 * (bounds_high - bounds_low)
    mutation_min      = 0.01 * (bounds_high - bounds_low)
    c1_init, c2_init  = 1.5, 1.5
    w_inertia_init    = 0.4
    wg_init           = 0.5

    pop  = rng.uniform(bounds_low, bounds_high, (pop_size, 2 * sensor_count))
    vel  = np.zeros_like(pop)

    def obj(a):
        return (alpha_weight * static_metrics(a.reshape(-1, 2))[5] +
                (1 - alpha_weight) * robust_penalty(a, rng, mc_train))

    pbest        = pop.copy()
    pbest_scores = np.array([obj(a) for a in pop])
    gbest_idx    = np.argmin(pbest_scores)

    # simple pack state
    pack_of = np.repeat(np.arange(pack_count_init), np.ceil(pop_size/pack_count_init))[:pop_size]
    improvement_hist = np.zeros((pop_size,), dtype=float)

    converg = []; best_so_far = np.inf

    def diversity(pop): return np.trace(np.cov(pop.T))
    def entropy_controller(div):
        div_norm = np.clip((div - 1e-6) / (1e6 - 1e-6 + 1e-9), 0, 1)
        mut_amp  = mutation_min + (1 - div_norm) * (mutation_max - mutation_min)
        wg       = wg_init * (0.5 + 0.5 * (1 - div_norm))
        return mut_amp, wg

    def select_leaders_and_weights(pop, pbest, pbest_scores, pack_of, k_lead=3):
        k = pack_of.max() + 1
        L, W = [], np.zeros((k, 3))
        for pid in range(k):
            idx = np.where(pack_of == pid)[0]
            if len(idx) == 0: L.append(None); continue
            top_idx = idx[np.argsort(pbest_scores[idx])[:min(k_lead, len(idx))]]
            leaders = pbest[top_idx]
            contrib = improvement_hist[top_idx]
            if contrib.sum() > 0: wts = contrib / contrib.sum()
            else: wts = np.ones(len(top_idx)) / len(top_idx)
            if len(top_idx) < 3:
                wts = np.pad(wts, (0, 3 - len(top_idx)), constant_values=0)
                leaders = np.vstack([leaders, np.repeat(leaders[-1:], 3 - len(top_idx), axis=0)])
            L.append(leaders[:3]); W[pid] = wts[:3]
        return L, W

    def gwo_pack_pull(xi, leaders, wts, a):
        A = 2 * a * rng.random(xi.shape[0]) - a
        C = 2 * rng.random(xi.shape[0])
        acc = np.zeros_like(xi)
        for j in range(3):
            acc += wts[j] * (leaders[j] - A * np.abs(C * leaders[j] - xi))
        return acc

    def re_cluster_packs(pop, scores, pack_count):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=pack_count, n_init='auto', random_state=0)
        return km.fit_predict(pop)

    def assign_roles(pop, scores, t):
        roles = np.zeros(pop.shape[0], dtype=int)
        q1 = max(1, int(0.2 * pop.shape[0])); order = np.argsort(scores)
        roles[order[:q1]]    = 1
        roles[order[q1:-q1]] = 2
        roles[order[-q1:]]   = 0
        return roles

    def migrate(pop, vel, pbest, pbest_scores, pack_of, q=2):
        k = pack_of.max() + 1
        for pid in range(k):
            idx = np.where(pack_of == pid)[0]
            if len(idx) <= q: continue
            worst = idx[np.argsort(pbest_scores[idx])[-q:]]
            pop[worst] = pbest[gbest_idx] + rng.normal(0, 0.01, size=(q, pop.shape[1]))
            vel[worst] *= 0
        return pop, vel

    roles = assign_roles(pop, pbest_scores, 0)

    for t in range(iterations + 1):
        gbest = pbest[gbest_idx]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(-1, 2)[alive_mask[:gbest.reshape(-1,2).shape[0]]]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break

        div = diversity(pop)
        mut_amp, wg_raw = entropy_controller(div)
        a  = 2 - 2 * t / iterations
        wg = (0.5 - 0.4 * t / iterations) * wg_raw
        c1, c2 = c1_init, c2_init; w_inertia = w_inertia_init

        if t % 20 == 0 and t > 0: pack_of = re_cluster_packs(pop, pbest_scores, pack_count_init)
        if t % 15 == 0: roles = assign_roles(pop, pbest_scores, t)
        leaders, leader_wts = select_leaders_and_weights(pop, pbest, pbest_scores, pack_of)
        if t % 25 == 0 and t > 0: pop, vel = migrate(pop, vel, pbest, pbest_scores, pack_of, q=2)

        for i in range(pop.shape[0]):
            r1, r2 = rng.random(pop.shape[1]), rng.random(pop.shape[1])
            vel[i] = (w_inertia * vel[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (pbest[gbest_idx] - pop[i]))
            pid = pack_of[i]
            if leaders[pid] is not None:
                Xg = gwo_pack_pull(pop[i], leaders[pid], leader_wts[pid], a)
                vel[i] += wg * (Xg - pop[i])
            if roles[i] == 0:
                pop[i] += vel[i] + rng.normal(0, mut_amp, size=pop.shape[1])
            elif roles[i] == 1:
                pop[i] += vel[i] + rng.normal(0, 0.2 * mut_amp, size=pop.shape[1])
            else:
                pop[i] += vel[i]
            pop[i] = np.clip(pop[i], bounds_low, bounds_high)

        scores = np.array([
            (alpha_weight * static_metrics(a.reshape(-1, 2))[5] +
             (1 - alpha_weight) * robust_penalty(a, rng, mc_train))
            for a in pop
        ])
        improved = scores < pbest_scores
        improvement_hist[improved] += (pbest_scores[improved] - scores[improved])
        pbest[improved]        = pop[improved]
        pbest_scores[improved] = scores[improved]
        gbest_idx = np.argmin(pbest_scores)

    return pbest[gbest_idx], converg


def run_robust_pso_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, 2 * sensor_count))
    vel = np.zeros_like(pop)
    pbest        = pop.copy()
    pbest_scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
    gbest_idx    = np.argmin(pbest_scores)
    converg = []; best_so_far = np.inf
    for t in range(iterations+1):
        gbest = pbest[gbest_idx]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(-1,2)[alive_mask[:gbest.reshape(-1,2).shape[0]]]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break
        for i in range(pop_size):
            r1, r2 = rng.random(), rng.random()
            vel[i] = (0.7*vel[i] + 1.5*r1*(pbest[i] - pop[i]) + 1.5*r2*(pbest[gbest_idx] - pop[i]))
            pop[i]  = np.clip(pop[i] + vel[i], bounds_low, bounds_high)
            sc = robust_penalty(pop[i], rng, mc_train)
            if sc < pbest_scores[i]: pbest_scores[i] = sc; pbest[i] = pop[i].copy()
        gbest_idx = np.argmin(pbest_scores)
    return pbest[gbest_idx], converg


def run_robust_gwo_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, 2 * sensor_count))
    converg = []; best_so_far = np.inf
    for t in range(iterations+1):
        scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
        gbest   = pop[np.argmin(scores)]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(-1,2)[alive_mask[:gbest.reshape(-1,2).shape[0]]]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break
        idx = np.argsort(scores)
        alpha, beta, delta = pop[idx[:3]]
        a = 2 - 2*t/iterations
        new_pop = []
        for agent in pop:
            A = 2*a*rng.random(agent.shape[0]) - a
            C = 2*rng.random(agent.shape[0])
            X1 = alpha - A*np.abs(C*alpha - agent)
            X2 = beta  - A*np.abs(C*beta  - agent)
            X3 = delta - A*np.abs(C*delta - agent)
            new_pop.append((X1+X2+X3)/3.0)
        pop = np.clip(np.array(new_pop), bounds_low, bounds_high)
    scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
    return pop[np.argmin(scores)], converg


def run_robust_de_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, 2 * sensor_count))
    scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
    converg = []; best_so_far = np.inf
    for t in range(iterations+1):
        gbest = pop[np.argmin(scores)]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(-1,2)[alive_mask[:gbest.reshape(-1,2).shape[0]]]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break
        F, Cr = 0.8, 0.9
        for i in range(pop_size):
            idxs = rng.choice(pop_size, 3, replace=False)
            a, b, c = pop[idxs]
            donor = np.clip(a + F*(b - c), bounds_low, bounds_high)
            mask  = rng.random(pop.shape[1]) < Cr
            trial = np.where(mask, donor, pop[i])
            sc = robust_penalty(trial, rng, mc_train)
            if sc < scores[i]: pop[i] = trial; scores[i] = sc
    return pop[np.argmin(scores)], converg


def run_robust_ga_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, 2 * sensor_count))
    converg = []; best_so_far = np.inf
    for t in range(iterations+1):
        scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
        gbest = pop[np.argmin(scores)]
        alive_mask = sample_failures(rng, sensor_count)
        surv = gbest.reshape(-1,2)[alive_mask[:gbest.reshape(-1,2).shape[0]]]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break
        new = []
        while len(new) < pop_size:
            i1, i2 = rng.choice(pop_size, 2, replace=False)
            a, b = pop[i1], pop[i2]
            pt = rng.integers(1, pop.shape[1])
            o1 = np.concatenate([a[:pt], b[pt:]])
            o2 = np.concatenate([b[:pt], a[pt:]])
            mut_mask = rng.random(pop.shape[1]) < 0.01
            o1[mut_mask] = rng.uniform(bounds_low[mut_mask], bounds_high[mut_mask])
            mut_mask = rng.random(pop.shape[1]) < 0.01
            o2[mut_mask] = rng.uniform(bounds_low[mut_mask], bounds_high[mut_mask])
            new.extend([o1, o2])
        pop = np.array(new[:pop_size])
    scores = np.array([robust_penalty(a, rng, mc_train) for a in pop])
    return pop[np.argmin(scores)], converg


def run_robust_woa_with_convergence(rng: np.random.Generator):
    pop = rng.uniform(bounds_low, bounds_high, (pop_size, 2 * sensor_count))
    best = pop[0].copy()
    best_score = robust_penalty(best, rng, mc_train)
    converg = []; best_so_far = np.inf
    for t in range(iterations+1):
        alive_mask = sample_failures(rng, sensor_count)
        surv = best.reshape(-1,2)[alive_mask[:best.reshape(-1,2).shape[0]]]
        fit_after = static_metrics(surv)[5]
        if fit_after < best_so_far: best_so_far = fit_after
        converg.append(best_so_far)
        if t == iterations: break
        a = 2 - 2*t/iterations
        for i in range(pop_size):
            sc = robust_penalty(pop[i], rng, mc_train)
            if sc < best_score: best_score = sc; best = pop[i].copy()
        for i in range(pop_size):
            r = rng.random()
            A = 2*a*rng.random(pop.shape[1]) - a
            C = 2*rng.random(pop.shape[1])
            if r < 0.5:
                pop[i] = best - A*np.abs(C*best - pop[i])
            else:
                l = rng.uniform(-1,1,pop.shape[1])
                pop[i] = np.abs(best - pop[i]) * np.exp(l) * np.cos(2*np.pi*l) + best
        pop = np.clip(pop, bounds_low, bounds_high)
    return best, converg

# ---------------------------------------------------------------------
# Runner dict
# ---------------------------------------------------------------------
runners = {
    'DMPA‑GWO': run_mo_pso_dmpa_gwo_pp_with_convergence,
    'PSO':      run_robust_pso_with_convergence,
    'GWO':      run_robust_gwo_with_convergence,
    'DE':       run_robust_de_with_convergence,
    'GA':       run_robust_ga_with_convergence,
    'WOA':      run_robust_woa_with_convergence,
}

# ---------------------------------------------------------------------
# Evaluate all algorithms (single run each)
# ---------------------------------------------------------------------
def evaluate_all_algorithms(rng: np.random.Generator):
    results = {}
    child_seeds = rng.integers(0, 2**32-1, size=len(runners))
    for (name, fn), seed in zip(runners.items(), child_seeds):
        sub_rng = np.random.default_rng(seed)
        best_agent, converg = fn(sub_rng)
        sensors = best_agent.reshape(-1,2)

        # sample failure for baseline
        alive_mask = sample_failures(sub_rng, sensors.shape[0])
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

# ------------------ Persistence helpers (load only) -------------------
def _iter_runs():
    for p in PERSIST_DIR.glob("run_*"):
        meta = p / "meta.json"
        if meta.exists():
            try:
                info = json.loads(meta.read_text())
                yield p, info
            except Exception:
                continue

# Keys that define a solution (UI-only flags like auto_apply_relays are ignored)
MATCH_KEYS = (
    "env_source","area_size","sensor_count","sensor_range","comm_factor","obs_radius",
    "iterations","pop_size","alpha_weight","mc_train","mc_report",
    "failure_prob","relay_budget","rng_seed"
)
EPS = 1e-9

def _num_equal(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=EPS)
    except Exception:
        return False

def _val_equal(a, b):
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        return _num_equal(a, b)
    return a == b

def _params_match(saved: dict, current: dict) -> bool:
    for k in MATCH_KEYS:
        if k not in saved or k not in current:
            return False
        if not _val_equal(saved[k], current[k]):
            return False
    return True

def find_saved_run_for_params(params: dict):
    candidates = []
    for p, info in _iter_runs():
        saved = info.get("params", {})
        if _params_match(saved, params):                  # fuzzy/partial match
            candidates.append((p, info, p.stat().st_mtime))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0], candidates[0][1]

def find_latest_default_run():
    candidates = []
    for p, info in _iter_runs():
        prm = info.get("params", {})
        if prm.get("env_source") == "default_or_synthetic":
            candidates.append((p, info, p.stat().st_mtime))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0], candidates[0][1]

def load_results_from_dir(run_dir: Path):
    with (run_dir / "results.pkl").open("rb") as f:
        return pickle.load(f)

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
    sensors = np.asarray(sensors).reshape(-1, 2)
    alive_mask = np.asarray(alive_mask).astype(bool)[:sensors.shape[0]]
    G_base = build_comm_graph(sensors[alive_mask], comm_range)
    if relays.size > 0:
        all_pts = np.vstack([sensors[alive_mask], relays])
        G = build_comm_graph(all_pts, comm_range)
    else:
        G = G_base
        all_pts = sensors[alive_mask]
    mst = nx.minimum_spanning_tree(G)
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_title("Failure Scenario Explorer", fontsize=14, weight='bold')
    for x,y,r in obstacles:
        ax.add_patch(Circle((x,y), r, edgecolor='red', facecolor='red', alpha=0.3))
    for (x,y), alive in zip(sensors, alive_mask):
        ax.add_patch(Circle((x,y), sensor_range,
                            edgecolor='purple' if alive else 'gray',
                            facecolor='purple' if alive else 'none',
                            alpha=0.10 if alive else 0.0, linewidth=0.5))
        ax.add_patch(Circle((x,y), comm_range,
                            edgecolor='cyan', facecolor='none',
                            alpha=0.5 if alive else 0.15, linestyle='--', linewidth=0.8))
    for x,y in relays:
        ax.add_patch(Circle((x,y), sensor_range,
                            edgecolor='orange', facecolor='orange', alpha=0.15, linewidth=0.5))
        ax.add_patch(Circle((x,y), comm_range,
                            edgecolor='orange', facecolor='none', linestyle=':', alpha=0.6, linewidth=0.8))
    pts = all_pts
    for u,v in mst.edges():
        x0,y0 = pts[u]; x1,y1 = pts[v]
        ax.plot([x0,x1],[y0,y1],color='blue',linewidth=2)
    surv_pts = sensors[alive_mask]
    ax.scatter(surv_pts[:,0], surv_pts[:,1], s=80, edgecolor='white', facecolor='none', linewidth=1.5, label='Alive')
    fail_pts = sensors[~alive_mask]
    if fail_pts.size>0:
        ax.scatter(fail_pts[:,0], fail_pts[:,1], marker='x', c='red', s=100, linewidths=2, label='Failed')
    if relays.size>0:
        ax.scatter(relays[:,0], relays[:,1], marker='*', c='orange', s=140, edgecolor='black', linewidths=0.8, label='Relays')
    ax.scatter(targets[:,0], targets[:,1], c='green', s=40, label='Targets')
    ax.set_xlim(0, area_size); ax.set_ylim(0, area_size)
    ax.set_aspect('equal'); ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------
# Multi‑Ring Before/After Plot
# ---------------------------------------------------------------------
def plot_before_after(name, sensors, alive_mask, comm_range, obstacles, targets,
                      sensor_range, area_size):
    sensors = np.asarray(sensors).reshape(-1, 2)
    alive_mask = np.asarray(alive_mask).astype(bool)[:sensors.shape[0]]

    G_full = build_comm_graph(sensors, comm_range)
    mst_full = nx.minimum_spanning_tree(G_full)
    surv_idx = np.where(alive_mask)[0]
    G_surv = G_full.subgraph(surv_idx)
    mst_surv = nx.minimum_spanning_tree(G_surv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle(f"{name}: Before vs After Failure", fontsize=14, weight='bold')

    def _draw(ax, title, mst_edges, show_failed=False):
        ax.set_title(title)
        for x, y, r in obstacles:
            ax.add_patch(Circle((x, y), r, edgecolor='darkgreen', facecolor='darkgreen', alpha=0.3))
        for x, y in sensors:
            ax.add_patch(Circle((x, y), sensor_range,
                                edgecolor='forestgreen', facecolor='forestgreen',
                                alpha=0.15, linewidth=0.5))
            ax.add_patch(Circle((x, y), comm_range,
                                edgecolor='limegreen', facecolor='none',
                                alpha=0.5, linestyle='--', linewidth=0.8))
        edge_color = 'seagreen' if show_failed else 'lightgreen'
        for u, v in mst_edges.edges():
            ax.plot([sensors[u, 0], sensors[v, 0]],
                    [sensors[u, 1], sensors[v, 1]],
                    color=edge_color, linewidth=2)
        if show_failed:
            surv = sensors[alive_mask]
            ax.scatter(surv[:, 0], surv[:, 1],
                       s=60, edgecolor='forestgreen', facecolor='none',
                       linewidth=1.5, label='Sensors')
            fail = sensors[~alive_mask]
            if fail.size > 0:
                ax.scatter(fail[:, 0], fail[:, 1],
                           marker='x', c='red', s=80, linewidths=2, label='Failed')
        else:
            ax.scatter(sensors[:, 0], sensors[:, 1],
                       s=60, edgecolor='forestgreen', facecolor='none', linewidth=1.5, label='Sensors')
        ax.scatter(targets[:, 0], targets[:, 1],
                   c='darkolivegreen', s=40, label='Targets')
        ax.set_xlim(0, area_size); ax.set_ylim(0, area_size)
        ax.set_aspect('equal'); ax.grid(True, linestyle='--', alpha=0.3)

    _draw(ax1, "Before Failure", mst_full, show_failed=False)
    _draw(ax2, "After Failure", mst_surv, show_failed=True)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkolivegreen', markersize=6, label='Targets'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='forestgreen', markersize=6, label='Sensors'),
        Line2D([0], [0], marker='x', color='red', markersize=8, label='Failed'),
        Patch(facecolor='darkgreen', edgecolor='darkgreen', alpha=0.3, label='Obstacles'),
        Patch(facecolor='forestgreen', edgecolor='forestgreen', alpha=0.15, label='Sense Range'),
        Line2D([0], [0], color='limegreen', linestyle='--', label='Comm Range'),
    ]
    ax2.legend(handles=handles, loc='upper right')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# ---------------------------------------------------------------------
# Main Execution Flow w/ Session State Persistence + Fuzzy load
# ---------------------------------------------------------------------
st.title("📡 Sensor Deployment Optimization Dashboard")

# Build a parameter signature so we know when to invalidate cached results.
param_sig = (
    area_size, sensor_count, sensor_range, comm_factor, obs_radius,
    iterations, pop_size, alpha_weight, mc_train, mc_report,
    failure_prob, relay_budget, fail_scenarios, rng_seed, auto_apply_relays
)

def build_param_bundle(env_source: str):
    return dict(
        area_size=float(area_size), sensor_count=int(sensor_count), sensor_range=float(sensor_range),
        comm_factor=float(comm_factor), obs_radius=float(obs_radius),
        iterations=int(iterations), pop_size=int(pop_size), alpha_weight=float(alpha_weight),
        mc_train=int(mc_train), mc_report=int(mc_report), failure_prob=float(failure_prob),
        relay_budget=int(relay_budget), rng_seed=int(rng_seed),
        env_source=env_source,
        # UI-only params (ignored by matching) can be stored but aren't used for equality
        auto_apply_relays=bool(auto_apply_relays)
    )

# Decide env_source
env_source = "uploaded" if uploaded is not None else "default_or_synthetic"
current_params = build_param_bundle(env_source=env_source)

results = None
loaded_from_persist = False

# --------- Load from .persist first (fuzzy/partial exact match) ---------
if uploaded is None:
    exact_path, _ = find_saved_run_for_params(current_params)
    if exact_path:
        results = load_results_from_dir(exact_path)
        loaded_from_persist = True
        try: st.toast(f"Loaded saved results (match): {exact_path.name}", icon="📦")
        except: st.info(f"Loaded saved results (match): {exact_path.name}")
    else:
        latest_path, _ = find_latest_default_run()
        if latest_path:
            results = load_results_from_dir(latest_path)
            loaded_from_persist = True
            try: st.toast(f"Loaded latest default results: {latest_path.name}", icon="📦")
            except: st.info(f"Loaded latest default results: {latest_path.name}")
            st.caption("Showing latest saved default run; parameters may differ. Click **Run Optimization** to recompute with current sliders.")

# ------------------ Clear recompute guard ------------------
should_recompute = (
    run_button
    or (results is None)                                 # nothing loaded
    or (uploaded is not None and st.session_state.get('_param_sig') != param_sig)
)

if should_recompute:
    with st.spinner("Running optimizations… (this can take a minute)"):
        results = evaluate_all_algorithms(master_rng)

    # Update session state so the UI uses this recomputed data
    st.session_state['results']    = results
    st.session_state['_param_sig'] = param_sig
    # Failure scenarios for explorer (use current slider sensor_count)
    st.session_state['fail_masks'] = precompute_failure_scenarios(
        fail_scenarios, rng_seed+2025, int(sensor_count), failure_prob
    )

    # Do NOT save recomputed results unless explicitly enabled
    if SAVE_NEW_RUNS:
        pass
    else:
        try: st.toast("Recomputed results (not saved).", icon="⚠️")
        except: st.info("Recomputed results (not saved).")
else:
    # Loaded from persist; ensure session_state is set
    if loaded_from_persist:
        st.session_state['results']    = results
        st.session_state['_param_sig'] = param_sig
        if 'fail_masks' not in st.session_state:
            st.session_state['fail_masks'] = None

# Pull from session_state
results = st.session_state.get('results', None)
if results is None:
    st.info("Configure parameters in the sidebar and click **Run Optimization**.")
    st.stop()

# ---------------- Convergence data ----------------
conv_raw = {name: np.asarray(data['convergence'], dtype=float) for name, data in results.items()}
max_len = max(len(v) for v in conv_raw.values())
conv_padded = {name: np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for name, v in conv_raw.items()}
conv_df = pd.DataFrame(conv_padded); conv_df.index.name = "Iteration"
bsf_df = conv_df.cummin(skipna=True)

st.subheader("Convergence Comparison (Best‑so‑Far After‑Failure Static Fitness)")
st.line_chart(bsf_df)

# ---------------- Before/After Plots ----------------
st.subheader("Before vs After Failure Deployments")
for name in runners.keys():
    data = results[name]
    sensors = np.asarray(data['best_agent']).reshape(-1,2)  # robust reshape
    alive_mask = np.asarray(data['sample_alive_mask'])
    if alive_mask.shape[0] != sensors.shape[0]:
        tmp_rng = np.random.default_rng(rng_seed + hash(name) % 100000)
        alive_mask = sample_failures(tmp_rng, sensors.shape[0])
        data['sample_alive_mask'] = alive_mask
    fig = plot_before_after(name, sensors, alive_mask,
                            comm_range, obstacles, targets,
                            sensor_range, area_size)
    st.pyplot(fig)

# ---------------- DMPA‑GWO Relay Analysis ----------------
st.subheader("DMPA‑GWO Relay Repair Analysis")
dm_data    = results['DMPA‑GWO']
dm_sensors = np.asarray(dm_data['best_agent']).reshape(-1,2)
dm_alive   = np.asarray(dm_data['sample_alive_mask'])
if dm_alive.shape[0] != dm_sensors.shape[0]:
    tmp_rng = np.random.default_rng(rng_seed + 2025)
    dm_alive = sample_failures(tmp_rng, dm_sensors.shape[0])
    dm_data['sample_alive_mask'] = dm_alive
dm_surv    = dm_sensors[dm_alive]

cov_b, ov_b, conn_b, ovl_b, bnd_b, sf_b = dm_data['metrics_before']
cov_a, ov_a, conn_a, ovl_a, bnd_a, sf_a = dm_data['metrics_after_sample']

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

fig_rel = explorer_plot(dm_sensors, dm_alive, relays, comm_range, obstacles, targets, sensor_range, area_size)
st.pyplot(fig_rel)

# ---------------- Interactive Failure Scenario Explorer ----------------
st.subheader("🔍 Failure Scenario Explorer (DMPA‑GWO)")
n_loaded = dm_sensors.shape[0]
fail_masks = st.session_state.get('fail_masks')
if (fail_masks is None or
    fail_masks.shape[0] != fail_scenarios or
    fail_masks.shape[1] != n_loaded):
    fail_masks = precompute_failure_scenarios(fail_scenarios, rng_seed+2025, n_loaded, failure_prob)
    st.session_state['fail_masks'] = fail_masks

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

fig_expl = explorer_plot(dm_sensors, scen_alive, relays_s, comm_range, obstacles, targets, sensor_range, area_size)
st.pyplot(fig_expl)

# ---------------- Summary Table ----------------
st.subheader("📋 Summary Table")
final_bsf = bsf_df.iloc[-1]

rows = []
for name, data in results.items():
    covB, ovB, connB, ovlB, bndB, sfB = data['metrics_before']
    covA, ovA, connA, ovlA, bndA, sfA = data['metrics_after_sample']
    mc = data['metrics_mc']
    if name == 'DMPA‑GWO' and auto_apply_relays:
        covA, ovA, connA, ovlA, bndA, sfA = cov_r, ov_r, conn_r, ovl_r, bnd_r, sf_r
    rows.append({
        "Algorithm":                name,
        "Final Conv (best‑so‑far)": final_bsf[name],
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





# ---------------- Parameter Guide (wrapped, no truncation) ----------------
st.subheader("ℹ️ Parameter Guide")

# ---------------- Summary Column Guide (wrapped, no truncation) ----------------
with st.expander("ℹ️ Summary Table — column meanings", expanded=False):
    col_desc = {
        "Final Conv (best-so-far)":
            "Last value of the optimizer’s best-so-far after-failure static fitness trace (uses a per-iteration sampled failure). "
            "This is a convergence indicator, not the MC average. Lower is better.",
        "StaticFit B→A":
            "Static fitness before failure → after one sampled failure (for DMPA-GWO this may be the relay-"
            "repaired 'after' if Auto-apply relays is ON). Lower (more negative) is better. "
            "Formula: −1.5·coverage + 0.01·overlap_pairs + 0.01·boundary_out + 0.5·(1−connectivity) + 0.5·over_coverage_rate.",

        "ExpFit After (MC)":
            "Expected (Monte-Carlo averaged) static fitness after failures over the reporting MC samples. Lower is better.",

        "Fit Std (MC)":
            "Standard deviation of the MC fitness values (robustness/variability). Lower is better.",

        "Coverage B→A":
            "Fraction of targets covered by ≥1 sensor before → after the sampled failure (or repaired for "
            "DMPA-GWO if Auto-apply relays is ON). Range 0–1; higher is better.",

        "ExpCov After (MC)":
            "Expected coverage after failures, averaged across MC trials. Range 0–1; higher is better.",

        "Cov Std (MC)":
            "Standard deviation of coverage over MC trials. Lower is better.",

        "Connectivity B→A":
            "Fraction of surviving sensors connected (via edges ≤ sensor_range) to the 'base' region "
            "(sensors within sensor_range of the area center), before → after the sampled failure. Range 0–1; higher is better.",

        "ExpConn After (MC)":
            "Expected connectivity after failures, averaged across MC trials. Range 0–1; higher is better.",

        "Conn Std (MC)":
            "Standard deviation of connectivity over MC trials. Lower is better.",

        "OverCov B→A":
            "Over-coverage rate (fraction of targets covered by more than one sensor) before → after the sampled failure "
            "(or repaired for DMPA-GWO if Auto-apply relays is ON). Range 0–1.",

        "Overlap B→A":
            "Count of sensor pairs closer than sensor_range/2 (crude overlap/congestion) before → after. Integer; lower is better.",

        "Boundary B→A":
            "Count of sensors outside the [0, area_size] square before → after. Integer; lower is better.",

        
    }

    for name, desc in col_desc.items():
        st.markdown(f"**{name}**  \n{desc}")
        st.divider()

    st.caption(
        "Notes: 'B→A' columns compare a single snapshot (Before vs one After failure). "
        "'Exp… (MC)' columns are expectations across many Monte-Carlo failure draws."
    )
