# app.py
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ======================================================
# Fixed GA Parameters (as required by the question)
# ======================================================
POP_SIZE = 300            # Population size
CHROM_LEN = 80            # Chromosome length
TARGET_ONES = 40          # Fitness peaks at 40 ones
MAX_FITNESS = 80          # Maximum fitness value
N_GENERATIONS = 50        # Number of generations

# ======================================================
# GA Hyperparameters
# ======================================================
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.85
MUTATION_RATE = 1.0 / CHROM_LEN   # ~1 bit flip per chromosome

# ======================================================
# Fitness Function
# ======================================================
def fitness(individual: np.ndarray) -> float:
    """
    Fitness is maximized (80) when the number of 1s equals 40.
    """
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)

# ======================================================
# Genetic Algorithm Operators
# ======================================================
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

# ======================================================
# GA Evolution Process (with Elitism)
# ======================================================
def evolve(pop: np.ndarray, generations: int):
    best_fitness_history = []
    best_individual = None
    best_fitness = -np.inf

    for _ in range(generations):
        fits = np.array([fitness(ind) for ind in pop])

        # Track best individual
        gen_best_idx = np.argmax(fits)
        gen_best_f = fits[gen_best_idx]
        best_fitness_history.append(float(gen_best_f))

        if gen_best_f > best_fitness:
            best_fitness = float(gen_best_f)
            best_individual = pop[gen_best_idx].copy()

        # Elitism: keep the best individual
        new_pop = [best_individual.copy()]

        # Generate rest of population
        while len(new_pop) < len(pop):
            p1 = tournament_selection(pop, fits, TOURNAMENT_K)
            p2 = tournament_selection(pop, fits, TOURNAMENT_K)
            c1, c2 = single_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])

        pop = np.array(new_pop[:len(pop)], dtype=np.int8)

    return best_individual, best_fitness, best_fitness_history

# ======================================================
# Streamlit Web Application
# ======================================================
st.set_page_config(
    page_title="Genetic Algorithm – 80-bit Pattern",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Genetic Algorithm: 80-bit Binary Pattern Evolution")
st.caption(
    "Population = 300 | Chromosome Length = 80 | Target Ones = 40 | "
    "Max Fitness = 80 | Generations = 50"
)

with st.expander("ℹ️ Fixed Problem Parameters", expanded=True):
    st.markdown(f"""
- **Population size**: {POP_SIZE}  
- **Chromosome length**: {CHROM_LEN}  
- **Target number of ones**: {TARGET_ONES}  
- **Maximum fitness**: {MAX_FITNESS}  
- **Generations**: {N_GENERATIONS}  
- **Selection**: Tournament selection (k = {TOURNAMENT_K})  
- **Crossover**: Single-point crossover (rate = {CROSSOVER_RATE})  
- **Mutation**: Bit-flip mutation (rate = {MUTATION_RATE:.4f})
""")

st.divider()

col1, col2 = st.columns([1, 2])
with col1:
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
with col2:
    st.write("")
    run_btn = st.button("Run Genetic Algorithm", type="primary", use_container_width=True)

# ======================================================
# Run GA
# ======================================================
if run_btn:
    with st.spinner("Running Genetic Algorithm..."):
        random.seed(seed)
        np.random.seed(seed)

        population = init_population(POP_SIZE, CHROM_LEN)
        best_ind, best_fit, history = evolve(population, N_GENERATIONS)

    ones_count = int(best_ind.sum())
    zeros_count = CHROM_LEN - ones_count
    bitstring = "".join(map(str, best_ind.tolist()))

    st.success("Evolution completed successfully!")

    st.subheader("🏆 Best Individual Found")
    st.metric("Best Fitness", f"{best_fit:.0f} / {MAX_FITNESS}")

    cols = st.columns(3)
    cols[0].metric("Number of 1s", ones_count)
    cols[1].metric("Number of 0s", zeros_count)
    cols[2].metric("Chromosome Length", CHROM_LEN)

    st.text("Best Bit Pattern:")
    st.code(bitstring, language="text")

    if ones_count == TARGET_ONES and best_fit == MAX_FITNESS:
        st.balloons()
        st.success("Perfect solution achieved (40 ones, fitness = 80) ✅")
    else:
        st.info("Near-optimal solution found. Try another seed for different results.")

    # Convergence Plot
    st.subheader("📈 Fitness Convergence")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(history) + 1), history, linewidth=2)
    ax.axhline(MAX_FITNESS, linestyle="--", alpha=0.6, label="Maximum Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness per Generation")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

st.caption("Genetic Algorithm Web Application")
