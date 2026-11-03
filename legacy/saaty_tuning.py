# Оригінал: "Метод Сааті tuning.py"
import numpy as np
import pandas as pd
import pygad
import matplotlib.pyplot as plt

# ==========================================
# 1. ВХІДНІ ДАНІ (АВТОПІДПРИЄМСТВО)
# ==========================================

FCM_STRUCTURE = [
    {'src': 0, 'tgt': 5, 'init': 0.3333, 'min': 0.13, 'max': 0.53},
    {'src': 0, 'tgt': 6, 'init': 0.5556, 'min': 0.36, 'max': 0.76},
    {'src': 1, 'tgt': 5, 'init': 0.1111, 'min': 0.02, 'max': 0.31},
    {'src': 1, 'tgt': 6, 'init': 0.3333, 'min': 0.13, 'max': 0.53},
    {'src': 2, 'tgt': 5, 'init': -1.0000, 'min': -0.98, 'max': -0.8},
    {'src': 2, 'tgt': 6, 'init': -0.7500, 'min': -0.95, 'max': -0.55},
    {'src': 3, 'tgt': 7, 'init': -0.5000, 'min': -0.7, 'max': -0.3},
    {'src': 3, 'tgt': 8, 'init': 0.1111, 'min': 0.02, 'max': 0.31},
    {'src': 4, 'tgt': 0, 'init': 0.1111, 'min': 0.02, 'max': 0.31},
    {'src': 4, 'tgt': 1, 'init': 0.3333, 'min': 0.13, 'max': 0.53},
    {'src': 4, 'tgt': 2, 'init': -0.1250, 'min': -0.32, 'max': -0.02},
    {'src': 4, 'tgt': 7, 'init': -0.1250, 'min': -0.32, 'max': -0.02},
    {'src': 5, 'tgt': 8, 'init': 1.0000, 'min': 0.8, 'max': 0.98},
    {'src': 6, 'tgt': 8, 'init': 0.5556, 'min': 0.36, 'max': 0.76},
    {'src': 7, 'tgt': 3, 'init': 0.1111, 'min': 0.02, 'max': 0.31},
    {'src': 7, 'tgt': 4, 'init': 0.3333, 'min': 0.13, 'max': 0.53},
    {'src': 8, 'tgt': 9, 'init': 1.0000, 'min': 0.8, 'max': 0.98},
    {'src': 9, 'tgt': 7, 'init': 0.5556, 'min': 0.36, 'max': 0.76},
]

# Входи: x1-x5 (C1-C5). Виходи: C6-C10.
data_rows = [
    # No, C1,   C2,   C3,   C4,   C5,   C6,   C7,   C8,   C9,   C10
    [1, -0.8, -0.8,  0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8], # Найгірший
    [2, -0.6, -0.6,  0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6], # Дуже поганий
    [3, -0.3, -0.3,  0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3], # Поганий
    [4,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # Нейтральний (базовий)
    [5,  0.3,  0.3, -0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3], # Задовільний
    [6,  0.6,  0.6, -0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6], # Добрий
    [7,  0.8,  0.8, -0.8,  0.8,  0.8,  0.8,  0.8,  0.8,  0.8,  0.8], # Дуже добрий
    [8,  1.0,  1.0, -1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]  # Найкращий
]
cols = ['No', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
df_train = pd.DataFrame(data_rows, columns=cols)

print("\n=== Training data (from Article Tables 3 & 4) ===")
print(df_train.to_string(index=False))

TRAIN_DATA = df_train.drop(columns=['No']).values
NUM_CONCEPTS = 10
INPUT_INDICES = [0, 1, 2, 3, 4]  # C1-C5
OUTPUT_INDICES = [5, 6, 7, 8, 9]  # C6-C10

fitness_history = []


# ==========================================
# 2. ЛОГІКА МОДЕЛЮВАННЯ (FCM)
# ==========================================

def run_fcm_simulation(initial_vector, W, steps=50):
    curr = initial_vector.copy()
    delta = curr
    res = curr + np.matmul(delta, W)
    prev = curr
    curr = res

    for _ in range(steps - 1):
        delta = curr - prev
        nxt = curr + np.matmul(delta, W)
        if np.max(np.abs(nxt)) > 1e5: return nxt
        prev = curr
        curr = nxt
    return curr


def calculate_mse(solution_weights):
    W = np.zeros((NUM_CONCEPTS, NUM_CONCEPTS))
    for idx, conn in enumerate(FCM_STRUCTURE):
        W[conn['src'], conn['tgt']] = solution_weights[idx]

    total_squared_error = 0
    count = 0

    for row in TRAIN_DATA:
        init_vec = np.zeros(NUM_CONCEPTS)
        init_vec[INPUT_INDICES] = row[INPUT_INDICES]

        final_raw = run_fcm_simulation(init_vec, W, steps=50)

        # Нормалізація (Алгоритм 2.5)
        # Знаходимо theoretical min/max для поточної конфігурації (спрощено - беремо max abs)
        # В ідеалі тут має бути повний розрахунок меж, але для тюнінгу використовуємо max-scaling
        max_val = np.max(np.abs(final_raw))
        normalized = final_raw / max_val if max_val > 0 else final_raw

        outputs = normalized[OUTPUT_INDICES]
        targets = row[OUTPUT_INDICES]

        total_squared_error += np.sum((outputs - targets) ** 2)
        count += len(outputs)

    return total_squared_error / count


# ==========================================
# 3. НАЛАШТУВАННЯ PYGAD
# ==========================================

def fitness_func(ga_instance, solution, solution_idx):
    mse = calculate_mse(solution)
    if mse == 0: return 99999999
    return 1.0 / mse


def on_generation(ga_instance):
    best_sol_fitness = ga_instance.best_solution()[1]
    current_mse = 1.0 / best_sol_fitness if best_sol_fitness > 0 else 100
    fitness_history.append(current_mse)


gene_space = [{'low': c['min'], 'high': c['max']} for c in FCM_STRUCTURE]
initial_population = []
expert_genes = [c['init'] for c in FCM_STRUCTURE]
population_size = 50
initial_population.append(expert_genes)
for _ in range(population_size - 1):
    random_genes = [np.random.uniform(c['min'], c['max']) for c in FCM_STRUCTURE]
    initial_population.append(random_genes)

ga_instance = pygad.GA(
    num_generations=400,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=population_size,
    num_genes=len(FCM_STRUCTURE),
    gene_space=gene_space,
    initial_population=initial_population,
    parent_selection_type="rws",
    keep_parents=2,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10,
    on_generation=on_generation,
    random_seed=42
)

# ==========================================
# 4. ЗАПУСК І РЕЗУЛЬТАТИ
# ==========================================

print("Запуск Генетичного Алгоритму...")
ga_instance.run()

solution, solution_fitness, _ = ga_instance.best_solution()

# --- ГРАФІК ---
plt.figure(figsize=(12, 6), dpi=800)
plt.plot(fitness_history, linewidth=2.5, color='black')
plt.xlabel("")
plt.ylabel("")
axis_width = 2.0
plt.grid(False)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(axis_width)
ax.spines['left'].set_linewidth(axis_width)
plt.tick_params(axis='both', which='major', labelsize=16, width=axis_width)
plt.savefig('optimization_dynamics.tiff',
            dpi=800,
            format='tiff',
            bbox_inches='tight',
            pil_kwargs={"compression": "tiff_lzw"})
print("\n[INFO] Графік динаміки збережено.")

# --- ТАБЛИЦЯ 10 (Ваги) ---
print("\n=== Arc weights before and after tuning ===")
df_weights = []
for i, val in enumerate(solution):
    c = FCM_STRUCTURE[i]
    df_weights.append({
        'Link': f"w{c['src'] + 1},{c['tgt'] + 1}",
        'Before': c['init'],
        'Interval': f"[{c['min']}, {c['max']}]",
        'After': round(val, 4)
    })
print(pd.DataFrame(df_weights).to_string(index=False))

# --- ТАБЛИЦЯ 11 (Результати по C8) ---
print("\n=== Expert assessments and simulation results (Focus on C8 - Profit) ===")
TARGET_CONCEPT_INDEX = 7  # C8


def predict_outputs(w_vec, row_in):
    W = np.zeros((NUM_CONCEPTS, NUM_CONCEPTS))
    for idx, conn in enumerate(FCM_STRUCTURE):
        W[conn['src'], conn['tgt']] = w_vec[idx]
    init_vec = np.zeros(NUM_CONCEPTS)
    init_vec[INPUT_INDICES] = row_in[INPUT_INDICES]
    final = run_fcm_simulation(init_vec, W, steps=50)
    mx = np.max(np.abs(final))
    norm = final / mx if mx > 0 else final
    return norm[OUTPUT_INDICES]


df_results = []
init_genes = [c['init'] for c in FCM_STRUCTURE]

# Змінні для розрахунку загальних метрик (MSE та MAD)
total_sq_error_before = 0
total_abs_error_before = 0
total_sq_error_after = 0
total_abs_error_after = 0
total_elements = 0

for i, row in enumerate(TRAIN_DATA):
    targets_all = row[OUTPUT_INDICES]

    # Прогноз для всіх виходів
    outs_before = predict_outputs(init_genes, row)
    outs_after = predict_outputs(solution, row)

    # Розрахунок похибок для поточного сценарію
    sq_error_before = np.sum((outs_before - targets_all) ** 2)
    sq_error_after = np.sum((outs_after - targets_all) ** 2)

    abs_error_before = np.sum(np.abs(outs_before - targets_all))
    abs_error_after = np.sum(np.abs(outs_after - targets_all))

    # Накопичення глобальних змінних для Summary (без дублювання)
    total_sq_error_before += sq_error_before
    total_sq_error_after += sq_error_after

    total_abs_error_before += abs_error_before
    total_abs_error_after += abs_error_after

    total_elements += len(targets_all)

    df_results.append({
        'No': i + 1,
        'SSE Before': round(sq_error_before, 4),
        'SSE After': round(sq_error_after, 4),
        'Improvement': round(sq_error_before - sq_error_after, 4)
    })

# Виведення оновленої таблиці
print("\n=== Expert assessments and simulation results (SSE for all outputs) ===")
print(pd.DataFrame(df_results).to_string(index=False))

# --- SUMMARY METRICS (MSE & MAD) ---
mse_final_before = total_sq_error_before / total_elements
mad_final_before = total_abs_error_before / total_elements
mse_final_after = total_sq_error_after / total_elements
mad_final_after = total_abs_error_after / total_elements

print("\n=== Summary Metrics (Across all outputs C6-C10) ===")
print(f"MSE Before Tuning: {mse_final_before:.5f}")
print(f"MAD Before Tuning: {mad_final_before:.5f}")
print("-" * 30)
print(f"MSE After Tuning:  {mse_final_after:.5f}")
print(f"MAD After Tuning:  {mad_final_after:.5f}")
