import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Callable, Tuple, List

# -------------------- Problem Definitions --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  # 'bit' or 'real'
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]

def make_custom_bit_problem(length: int = 80, target_ones: int = 50) -> GAProblem:
    """
    Fitness peaks when number of ones = target_ones.
    Returns 80 at the peak.
    """
    def fitness(x: np.ndarray) -> float:
        n_ones = np.sum(x)
        return float(80 - abs(target_ones - n_ones))  # max = 80 when ones = 50

    return GAProblem(
        name=f"Custom Bit Problem ({length} bits, peak at {target_ones} ones)",
        chromosome_type="bit",
        dim=length,
        bounds=None,
        fitness_fn=fitness
    )

# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    if problem.chromosome_type == "bit":
        return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)
    else:
        lo, hi = problem.bounds
        return rng.uniform(lo, hi, size=(pop_size, problem.dim))

def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)

def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y

def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)

def run_ga(
    problem: GAProblem,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_k: int,
    elitism: int,
    seed: int | None,
    stream_live: bool = True,
):
    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    # Live UI containers
    chart_area = st.empty()
    best_area = st.empty()

    history_best: List[float] = []
    history_avg: List[float] = []
    history_worst: List[float] = []

    for gen in range(generations):
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))
        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        if stream_live:
            df = pd.DataFrame({
                "Best": history_best,
                "Average": history_avg,
                "Worst": history_worst,
            })
            chart_area.line_chart(df)
            best_area.markdown(f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.6f}**")

        # Elitism
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if E > 0 else np.empty((0, pop.shape[1]))
        elites_fit = fit[elite_idx].copy() if E > 0 else np.empty((0,))

        # Next generation
        next_pop: List[np.ndarray] = []
        while len(next_pop) < pop_size - E:
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        # Insert elites
        pop = np.vstack([np.array(next_pop), elites]) if E > 0 else np.array(next_pop)
        fit = evaluate(pop, problem)

    # Final metrics
    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])
    df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    return {
        "best": best,
        "best_fitness": best_fit,
        "history": df,
        "final_population": pop,
        "final_fitness": fit,
    }

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="GA Bit Pattern", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm - Custom Bit Pattern")
st.caption("Generates bitstrings of length 80 with maximum fitness = 80 when number of ones = 50.")

# GA parameters fixed
problem = make_custom_bit_problem(length=80, target_ones=50)
pop_size = 300
generations = 50
crossover_rate = 0.9
mutation_rate = 0.01
tournament_k = 3
elitism = 2
seed = 42

if st.button("Run GA"):
    result = run_ga(
        problem=problem,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_k=tournament_k,
        elitism=elitism,
        seed=seed,
        stream_live=True
    )

    st.subheader("Fitness Over Generations")
    st.line_chart(result["history"])

    st.subheader("Best Solution")
    st.write(f"Best fitness: {result['best_fitness']:.6f}")
    bitstring = ''.join(map(str, result["best"].astype(int).tolist()))
    st.code(bitstring, language="text")
    st.write(f"Number of ones: {int(np.sum(result['best']))} / {problem.dim}")

    st.subheader("Final Population (first 20 individuals)")
    nshow = min(20, result["final_population"].shape[0])
    df = pd.DataFrame(result["final_population"][:nshow])
    df["fitness"] = result["final_fitness"][:nshow]
    st.dataframe(df, use_container_width=True)
