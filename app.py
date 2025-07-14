# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from scipy.spatial.distance import cdist


if 'first_load' not in st.session_state:
    st.session_state.first_load = True

# --- App Config ---
st.set_page_config(layout="wide", page_title="Smart Agri CSV Project")
st.title("Smart-Agriculture Sensor Deployment Optimization")

# --- Sidebar: Upload environment CSV ---
uploaded = st.sidebar.file_uploader(
    "Upload environment CSV (columns: type (target/obstacle), x, y):", type=["csv"]
)
if uploaded:
    df_env = pd.read_csv(uploaded)
else:
    # st.sidebar.info("No file uploaded → using default `sample_env_large.csv`")
    df_env = pd.read_csv("sample_env_large.csv")

# Read environment
# df_env = pd.read_csv(uploaded)
targets = df_env[df_env['type']=='target'][['x','y']].values
obstacles = df_env[df_env['type']=='obstacle'][['x','y']].values

# --- Sidebar: Algorithm parameters ---
st.sidebar.header("Optimization Parameters")
area_size       = st.sidebar.number_input("Area Size", value=100, min_value=1)
sensor_count    = st.sidebar.number_input("Sensor Count", value=30, min_value=1)
sensor_range    = st.sidebar.number_input("Sensor Range", value=15, min_value=1)
obstacle_radius = st.sidebar.number_input("Obstacle Radius", value=8, min_value=1)
iterations      = st.sidebar.number_input("Iterations", value=50, min_value=1)
pop_size        = st.sidebar.number_input("Population Size", value=20, min_value=1)

# --- Core Fitness Function ---
def advanced_fitness(agent):
    sensors = agent.reshape((sensor_count, 2))
    dt = cdist(targets, sensors)
    coverage_mask = dt <= sensor_range
    cov_per_target = np.sum(coverage_mask, axis=1)
    coverage_score = np.mean(cov_per_target > 0)
    over_cov_score = np.mean(cov_per_target > 1)
    pairwise = cdist(sensors, sensors)
    overlap = int(np.sum((pairwise < (sensor_range/2)) & (pairwise > 0)) // 2)
    boundary = int(np.sum((sensors < 0) | (sensors > area_size)))
    base = np.array([area_size/2, area_size/2])
    base_dists = np.linalg.norm(sensors - base, axis=1)
    adjacency = pairwise <= sensor_range
    visited = set(np.where(base_dists <= sensor_range)[0])
    queue = list(visited)
    while queue:
        u = queue.pop(0)
        for v in np.where(adjacency[u])[0]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    connectivity = len(visited) / sensor_count
    fitness = (
        -1.5 * coverage_score
        + 0.01 * overlap
        + 0.01 * boundary
        + 0.5 * (1 - connectivity)
        + 0.5 * over_cov_score
    )
    return fitness, coverage_score, over_cov_score, connectivity, overlap, boundary

# --- GWO Routines ---
def traditional_gwo_run():
    dim = 2 * sensor_count
    pop = np.random.uniform(0, area_size, (pop_size, dim))
    best_score = float('inf'); best_agent = None
    convergence = []
    metrics = {}
    start = time.perf_counter()
    for it in range(iterations):
        a = 2 - 2 * it / iterations
        scores = []
        for agent in pop:
            vals = advanced_fitness(agent)
            score = vals[0]; scores.append(score)
            if score < best_score:
                best_score = score; best_agent = agent.copy()
                metrics = dict(zip(['coverage','over_coverage','connectivity','overlap','boundary'], vals[1:]))
        alpha, beta, delta = pop[np.argsort(scores)[:3]]
        new_pop = []
        for agent in pop:
            A = 2 * a * np.random.rand(3) - a
            C = 2 * np.random.rand(3)
            D_alpha = np.abs(C[0]*alpha - agent)
            D_beta  = np.abs(C[1]*beta  - agent)
            D_delta = np.abs(C[2]*delta - agent)
            X1 = alpha - A[0]*D_alpha
            X2 = beta  - A[1]*D_beta
            X3 = delta - A[2]*D_delta
            new_pop.append((X1+X2+X3)/3)
        pop = np.array(new_pop)
        convergence.append(best_score)
    metrics['runtime'] = time.perf_counter() - start
    return best_agent, best_score, convergence, metrics


def dmpa_gwo_run():
    dim = 2 * sensor_count
    pop = np.random.uniform(0, area_size, (pop_size, dim))
    momentum = np.zeros_like(pop)
    influence = np.ones(3)
    best_score = float('inf'); best_agent = None
    convergence = []
    metrics = {}
    start = time.perf_counter()
    for it in range(iterations):
        a = 2 - 2 * it / iterations
        scores = []
        for agent in pop:
            vals = advanced_fitness(agent)
            score = vals[0]; scores.append(score)
            if score < best_score:
                best_score = score; best_agent = agent.copy()
                metrics = dict(zip(['coverage','over_coverage','connectivity','overlap','boundary'], vals[1:]))
        alpha, beta, delta = pop[np.argsort(scores)[:3]]
        new_pop, new_mom = [], []
        for idx, agent in enumerate(pop):
            A = 2 * a * np.random.rand(3) - a
            C = 2 * np.random.rand(3)
            D_alpha = np.abs(C[0]*alpha - agent)
            D_beta  = np.abs(C[1]*beta  - agent)
            D_delta = np.abs(C[2]*delta - agent)
            X1 = alpha - A[0]*D_alpha
            X2 = beta  - A[1]*D_beta
            X3 = delta - A[2]*D_delta
            weighted = (influence[0]*X1 + influence[1]*X2 + influence[2]*X3)/np.sum(influence)
            vel = weighted - agent + 0.5*momentum[idx]
            new_pop.append(agent + vel); new_mom.append(vel)
        pop = np.array(new_pop); momentum = np.array(new_mom)
        convergence.append(best_score)
    metrics['runtime'] = time.perf_counter() - start
    return best_agent, best_score, convergence, metrics

# --- Plotting Function ---
def plot_layout(agent, title, color, score):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_facecolor('#f7f7f7')
    ax.scatter(targets[:,0], targets[:,1], c='green', s=15, label='Targets')
    for i_pt, o in enumerate(obstacles):
        ax.add_patch(patches.Circle(o, radius=obstacle_radius, color='salmon', alpha=0.4,
                                     label='Obstacles' if i_pt==0 else None))
    s_arr = agent.reshape((sensor_count,2))
    for i_s, s in enumerate(s_arr):
        ax.add_patch(patches.Circle(s, radius=sensor_range, color=color, alpha=0.2,
                                     label='Coverage' if i_s==0 else None))
    ax.scatter(s_arr[:,0], s_arr[:,1], c=color, s=40, edgecolors='white', linewidth=1.2, label='Sensors')
    ax.set_xlim(0,area_size); ax.set_ylim(0,area_size)
    ax.set_title(f"{title}\nScore={round(score,3)}")
    ax.legend(loc='upper right', fontsize='small', markerscale=0.7, framealpha=0.8)
    return fig

if st.session_state.first_load:
    run_opt = True
    st.session_state.first_load = False
else:
    run_opt = st.sidebar.button("Run Optimization")


# --- Run Optimization on Click ---
if run_opt:
    np.random.seed(42)
    g_agent, g_score, g_conv, g_metrics = traditional_gwo_run()
    d_agent, d_score, d_conv, d_metrics = dmpa_gwo_run()

    # Convergence
    st.subheader("Convergence Comparison")
    st.line_chart(pd.DataFrame({ 'Traditional GWO': g_conv, 'DMPA-GWO++': d_conv }))

    # Layouts side by side
    cols = st.columns(2)
    with cols[0]:
        st.pyplot(plot_layout(g_agent, "Traditional GWO Layout", '#1f77b4', g_score))
    with cols[1]:
        st.pyplot(plot_layout(d_agent, "DMPA-GWO++ Layout", '#9467bd', d_score))

    # Metric Comparison
    metrics_df = pd.DataFrame([
        {'Metric':'Coverage (%)','Traditional GWO':round(g_metrics['coverage']*100,2),'DMPA-GWO++':round(d_metrics['coverage']*100,2)},
        {'Metric':'Over-Coverage (%)','Traditional GWO':round(g_metrics['over_coverage']*100,2),'DMPA-GWO++':round(d_metrics['over_coverage']*100,2)},
        {'Metric':'Connectivity (%)','Traditional GWO':round(g_metrics['connectivity']*100,2),'DMPA-GWO++':round(d_metrics['connectivity']*100,2)},
        {'Metric':'Overlap Count','Traditional GWO':g_metrics['overlap'],'DMPA-GWO++':d_metrics['overlap']},
        {'Metric':'Boundary Violations','Traditional GWO':g_metrics['boundary'],'DMPA-GWO++':d_metrics['boundary']},
        {'Metric':'Runtime (s)','Traditional GWO':round(g_metrics['runtime'],3),'DMPA-GWO++':round(d_metrics['runtime'],3)},
        {'Metric':'Best Fitness','Traditional GWO':round(g_score,4),'DMPA-GWO++':round(d_score,4)}
    ])
    st.subheader("Metric Comparison")
    st.table(metrics_df)

    # Final Metrics Bar Chart
    st.subheader("Final Metric Comparison: Traditional GWO vs DMPA-GWO++")
    fig2, ax2 = plt.subplots(figsize=(12,5))
    plot_df = metrics_df.set_index('Metric')
    bars = plot_df.plot(kind='bar', ax=ax2)
    ax2.set_ylabel("Value")
    ax2.set_xticklabels(plot_df.index, rotation=45, ha='right')
    ax2.set_xlabel("")
    ax2.legend(title="")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    # Annotate bar values
    for p in ax2.patches:
        ax2.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center', va='bottom', fontsize='small'
        )
    st.pyplot(fig2)
    
    
    
    
