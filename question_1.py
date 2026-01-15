# app.py
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---- Fixed GA Parameters ----
POP_SIZE = 300           # Population = 300
CHROM_LEN = 80           # Chromosome Length = 80
TARGET_ONES = 40         # Fitness peaks at ones = 40
MAX_FITNESS = 80         # Max fitness = 80 (when ones == 40)
N_GENERATIONS = 50       # Generations = 50

# ---- GA Hyperparameters (simple & sensible) ----
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.85
MUTATION_RATE = 1.0 / CHROM_LEN   # ~1 bit flip expected per chromosome

# ---- Fitness Function ----
def fitness(individual: np.ndarray) -> float:
    """
    Fitness is maximum (80) when number of 1s == 40
    The farther from 40 ones, the lower the fitness
    """
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)

# ---- GA Operators ----
def init_population(pop_size: int, chrom_len: int) -> np.ndarray:
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)

def tournament_selection(pop: np.ndarray, fits: np.ndarray, k: int) -> np.ndarray:
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()

def single_point_crossover(p1: np.ndarray, p2: np.ndarray):
    if np.random.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    
    point = np.random.randint(1, CHROM_LEN)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def mutate(individual: np.ndarray) -> np.ndarray:
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    individual[mask] = 1 - individual[mask]
    return individual

def evolve(pop: np.ndarray, generations: int):
    best_fitness_history = []
    best_individual = None
    best_fitness = -np.inf

    for gen in range(generations):
        fits = np.array([fitness(ind) for ind in pop])

        # Track best this generation and overall
        gen_best_idx = np.argmax(fits)
        gen_best_f = fits[gen_best_idx]
        best_fitness_history.append(float(gen_best_f))

        if gen_best_f > best_fitness:
            best_fitness = float(gen_best_f)
            best_individual = pop[gen_best_idx].copy()

        # Elitism: keep the current best (simple one individual elitism)
        new_pop = [best_individual.copy()]

        # Create next generation
        while len(new_pop) < len(pop):
            p1 = tournament_selection(pop, fits, TOURNAMENT_K)
            p2 = tournament_selection(pop, fits, TOURNAMENT_K)
            c1, c2 = single_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:len(pop)], dtype=np.int8)

    return best_individual, best_fitness, best_fitness_history

# ---- Streamlit UI ----
st.set_page_config(
    page_title="Genetic Algorithm - 80-bit Pattern",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Genetic Algorithm: Evolve 80-bit Pattern")
st.caption("Population = 300 | Length = 80 | Target ones = 40 | Max fitness = 80 | 50 generations")

with st.expander("ℹ️ Problem setup (fixed by requirement)", expanded=True):
    st.write(
        f"""
- **Population size**: `{POP_SIZE}`  
- **Chromosome length**: `{CHROM_LEN}`  
- **Target number of ones**: `{TARGET_ONES}`  
- **Max fitness at optimum**: `{MAX_FITNESS}` (when ones = {TARGET_ONES})  
- **Generations**: `{N_GENERATIONS}`  
- **Selection**: Tournament (k={TOURNAMENT_K})  
- **Crossover**: Single-point (rate={CROSSOVER_RATE})  
- **Mutation**: Bit-flip (per-bit rate={MUTATION_RATE:.4f})
"""
    )

st.divider()

col1, col2 = st.columns([1, 2])
with col1:
    seed = st.number_input("Random seed (for reproducibility)", min_value=0, value=42, step=1)

with col2:
    st.write("")
    st.write("")
    run_btn = st.button("Run Genetic Algorithm", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("Evolving population for 50 generations..."):
        # Reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Run GA
        population = init_population(POP_SIZE, CHROM_LEN)
        best_ind, best_fit, history = evolve(population, N_GENERATIONS)

    # Results
    ones_count = int(best_ind.sum())
    zeros_count = CHROM_LEN - ones_count
    bitstring = "".join(map(str, best_ind))

    st.success("Evolution completed!")

    st.subheader("🏆 Best Solution Found")
    st.metric("Best Fitness", f"{best_fit:.0f} / 80", delta=None)

    cols = st.columns(3)
    cols[0].metric("Number of 1s", ones_count)
    cols[1].metric("Number of 0s", zeros_count)
    cols[2].metric("Chromosome Length", CHROM_LEN)

    st.text("Best bit pattern:")
    st.code(bitstring, language="text")

    if ones_count == TARGET_ONES and best_fit == MAX_FITNESS:
        st.balloons()
        st.success("★★★ Perfect solution found! (40 ones & fitness = 80) ★★★")
    elif ones_count == TARGET_ONES:
        st.info("Perfect number of ones achieved (40), but fitness slightly below max")
    else:
        st.info("Near-optimal solution found. Try different seed for better results.")

    # Convergence plot 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(history) + 1), history, 'b-', linewidth=2.5)
    ax.axhline(y=MAX_FITNESS, color='red', linestyle='--', alpha=0.6, label='Theoretical Maximum (80)')
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness Evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

st.caption("© 2025–2026 Simple Genetic Algorithm Demo — 80 bits, target 40 ones")