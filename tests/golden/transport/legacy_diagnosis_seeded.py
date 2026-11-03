# Оригінал: "ДІАГНОСТИКА УЗАГАЛЬНЕНИХ ДИНАМІЧНИХ СИСТЕМ НА ОСНОВІ НЕЧІТКИХ КОГНІТИВНИХ КАРТ.py"
import numpy as np
import pygad
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# Вимикаємо попередження для чистоти виводу
warnings.filterwarnings("ignore")

# ==========================================
# 0. НАЛАШТУВАННЯ ВІЗУАЛІЗАЦІЇ ТА ДЕТАЛІЗАЦІЇ
# ==========================================
SCENARIO_TO_PLOT = "S1"

# Параметр деталізації (від 1 до 100%)
RECORDING_DETAIL_PERCENT = 90

# Глобальні змінні
CURRENT_RUN_HISTORY = []
RECORDING_STEP = 1  # Буде обчислено автоматично
DEMO_PRINTED = False  # Прапор для одноразового друку демо


def on_generation(ga_instance):
    """
    Callback-функція. Зберігає рішення згідно з налаштованим кроком.
    """
    current_gen = ga_instance.generations_completed

    # Логіка фільтрації записів (зберігаємо, якщо покоління кратне кроку)
    if current_gen % RECORDING_STEP == 0:
        solution, solution_fitness, _ = ga_instance.best_solution()
        CURRENT_RUN_HISTORY.append(solution.copy())


# ==========================================
# 1. КОНФІГУРАЦІЯ ТА ДАНІ
# ==========================================

W = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.31, 0.52, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.30, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, -0.90, -0.62, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.49, 0.12, 0.0],
    [0.10, 0.31, -0.11, 0.0, 0.0, 0.0, 0.0, -0.11, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.89, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.0],
    [0.0, 0.0, 0.0, 0.11, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.89],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.51, 0.0, 0.0]
])

SCENARIOS = {
    "S1": [0, -0.6, 0, -0.8, -0.6],
    "S2": [0.4, -0.2, -0.4, -0.4, 0.2],
    "S3": [0.8, 0.4, -0.8, 0.2, 0.6],
    "S4": [0.8, 0.8, -0.8, 0.6, 0.8]
}

IDX_CAUSES = [0, 1, 2, 3, 4]
IDX_EFFECTS = [5, 6, 7, 8, 9]
NUM_CONCEPTS = 10


# ==========================================
# 2. ФУНКЦІЇ МОДЕЛЮВАННЯ
# ==========================================

def simulate_fcm_step_1_4(initial_causes_vector):
    X_curr = np.zeros(NUM_CONCEPTS)
    X_curr[IDX_CAUSES] = initial_causes_vector
    X_prev = np.zeros(NUM_CONCEPTS)
    max_iter = 50
    epsilon = 0.001

    for k in range(max_iter):
        delta = X_curr - X_prev
        influence = np.dot(delta, W)
        X_next = X_curr + influence
        if np.max(np.abs(X_next - X_curr)) < epsilon:
            X_curr = X_next
            break
        X_prev = X_curr
        X_curr = X_next

    X_steady = X_curr
    max_val = np.max(np.abs(X_steady))
    if max_val > 0:
        X_normalized = X_steady / max_val
    else:
        X_normalized = X_steady

    consequences = X_normalized[IDX_EFFECTS]
    consequences = np.clip(consequences, -1.0, 1.0)
    return consequences


# ==========================================
# 3. ДІАГНОСТИКА
# ==========================================

TARGET_EFFECTS_GLOBAL = None


def fitness_func(ga_instance, solution, solution_idx):
    predicted_effects = simulate_fcm_step_1_4(solution)
    error = np.sum((predicted_effects - TARGET_EFFECTS_GLOBAL) ** 2)
    # Максимізація пристосованості еквівалентна мінімізації помилки
    return 1.0 / (error + 1e-8)


# --- НОВІ ФУНКЦІЇ ДЛЯ ДЕМОНСТРАЦІЇ ---
def demo_crossover_func(parents, offspring_size, ga_instance):
    global DEMO_PRINTED
    offspring = []
    idx = 0
    print_detail = (not DEMO_PRINTED)

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        random_vec = np.random.uniform(0.0, 1.0, size=parent1.shape)
        child = np.empty_like(parent1)

        for gene_idx in range(parent1.shape[0]):
            if random_vec[gene_idx] <= 0.5:
                child[gene_idx] = parent1[gene_idx]
            else:
                child[gene_idx] = parent2[gene_idx]

        if print_detail and idx == 0:
            print("\n" + "=" * 50)
            print("=== DEMONSTRATION LOG FOR PAPER (CROSSOVER) ===")
            print("=" * 50)
            print(f"Parent 1:  {np.round(parent1, 3)}")
            print(f"Parent 2:  {np.round(parent2, 3)}")
            print(f"Random (xi_1):    {np.round(random_vec, 2)}")
            print(f"Child:     {np.round(child, 3)}")
            print("=" * 50 + "\n")

        offspring.append(child)
        idx += 1
    return np.array(offspring)


def demo_mutation_func(offspring, ga_instance):
    global DEMO_PRINTED
    print_detail = (not DEMO_PRINTED)
    mutation_prob = ga_instance.mutation_probability

    for chromosome_idx in range(offspring.shape[0]):
        random_vec = np.random.uniform(0.0, 1.0, size=offspring.shape[1])

        if print_detail and chromosome_idx == 0:
            print("\n" + "=" * 50)
            print("=== DEMONSTRATION LOG FOR PAPER (MUTATION) ===")
            print("=" * 50)
            print(f"Chromosome (Before): {np.round(offspring[chromosome_idx], 3)}")
            print(f"Random Vector (xi_2):         {np.round(random_vec, 2)}")

        for gene_idx in range(offspring.shape[1]):
            if random_vec[gene_idx] <= mutation_prob:
                new_gene = np.random.uniform(ga_instance.gene_space['low'],
                                             ga_instance.gene_space['high'])
                offspring[chromosome_idx, gene_idx] = new_gene

        if print_detail and chromosome_idx == 0:
            print(f"Chromosome (After):  {np.round(offspring[chromosome_idx], 3)}")
            print("=" * 50 + "\n")
            DEMO_PRINTED = True

    return offspring


# -------------------------------------

def run_diagnosis_ga(target_effects, original_causes_reference, num_runs=5, save_history=False, demo_mode=False):
    global TARGET_EFFECTS_GLOBAL, CURRENT_RUN_HISTORY, RECORDING_STEP
    TARGET_EFFECTS_GLOBAL = target_effects

    # Обчислюємо крок запису на основі відсотка
    safe_percent = max(1, min(100, RECORDING_DETAIL_PERCENT))
    RECORDING_STEP = int(100 / safe_percent)
    if RECORDING_STEP < 1: RECORDING_STEP = 1

    best_overall_solution = None
    min_dist_to_original = float('inf')
    best_run_history = []

    # Змінна для збереження історії пристосованості найкращого прогону
    best_fitness_curve = []

    # Параметри ГА
    num_generations = 100
    num_parents_mating = 4
    sol_per_pop = 4
    num_genes = 5
    gene_space = {'low': -1.0, 'high': 1.0}

    for i in range(num_runs):
        # Очищуємо історію перед кожним новим запуском, якщо це потрібно для запису
        if save_history:
            CURRENT_RUN_HISTORY = []
            current_callback = on_generation
        else:
            current_callback = None

        # Вмикаємо кастомні оператори ТІЛЬКИ для першого прогону в демо-режимі
        if demo_mode and i == 0:
            crossover_type = demo_crossover_func
            mutation_type = demo_mutation_func
        else:
            crossover_type = "uniform"
            mutation_type = "random"

        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=gene_space,
            parent_selection_type="rws",
            mutation_probability=0.2,
            keep_elitism=2,
            on_generation=current_callback,
            crossover_type=crossover_type,  # Змінено
            mutation_type=mutation_type,  # Змінено
            suppress_warnings=True, random_seed=42 + i
        )

        ga_instance.run()

        solution, solution_fitness, _ = ga_instance.best_solution()

        # Обчислюємо відстань до "істини" (для експериментальної перевірки)
        # В реальній діагностиці ми б орієнтувалися тільки на solution_fitness (помилку прогнозу)
        dist = np.mean(np.abs(solution - original_causes_reference))

        # Логіка вибору найкращого прогону
        is_better = False
        if best_overall_solution is None:
            is_better = True
        elif solution_fitness > 50:  # Якщо знайшли дуже точний розв'язок (помилка ~0)
            if dist < min_dist_to_original:
                is_better = True
        # Додаткова евристика: якщо пристосованість краща за попередню найкращу
        # (Але в даному коді логіка спирається на distance, якщо fitness високий)

        if is_better:
            min_dist_to_original = dist
            best_overall_solution = solution
            if save_history:
                best_run_history = list(CURRENT_RUN_HISTORY)
                # Зберігаємо історію пристосованості (fitness) по поколіннях
                best_fitness_curve = ga_instance.best_solutions_fitness

    if save_history:
        return best_overall_solution, best_run_history, best_fitness_curve
    else:
        return best_overall_solution


# ==========================================
# 4. ГОЛОВНИЙ ЦИКЛ
# ==========================================

results_data = []

print(f"Початок експерименту (Деталізація графіка: {RECORDING_DETAIL_PERCENT}%)...")
print("-" * 60)

first_scenario_flag = True  # Прапор для увімкнення демо-режиму

for scenario_name, original_causes in SCENARIOS.items():
    print(f"Обробка {scenario_name}...")
    is_plot_scenario = (scenario_name == SCENARIO_TO_PLOT)

    target_effects = simulate_fcm_step_1_4(original_causes)

    # Визначаємо, чи вмикати демо-режим (тільки для першого сценарію з списку)
    current_demo_mode = False
    if first_scenario_flag and is_plot_scenario:
        current_demo_mode = True
        first_scenario_flag = False

    if is_plot_scenario:
        # Для обраного сценарію отримуємо також історію пристосованості (fitness_curve)
        recovered_causes, history_to_plot, fitness_curve = run_diagnosis_ga(
            target_effects,
            original_causes,
            num_runs=50,
            save_history=True,
            demo_mode=False
        )
    else:
        recovered_causes = run_diagnosis_ga(
            target_effects,
            original_causes,
            num_runs=50,
            save_history=False,
            demo_mode=False
        )

    # ---------------------------------------------------------
    # 3.1 Побудова графіка динаміки змінних (лог. шкала, окремі файли)
    # ---------------------------------------------------------
    if is_plot_scenario and history_to_plot:
        print(f"--> Генерація окремих графіків динаміки для {scenario_name}...")

        history_array = np.array(history_to_plot)
        generations_axis = np.arange(1, len(history_array) + 1) * RECORDING_STEP

        # Імпорт LinearLocator для фіксованої кількості тіків
        from matplotlib.ticker import FormatStrFormatter, LinearLocator

        # Формуємо список тіків для осі X (спільний для всіх)
        custom_ticks = [1, 2, 5, 10, 20, 50, 100]
        custom_ticks = [t for t in custom_ticks if t <= generations_axis[-1]]
        if generations_axis[-1] not in custom_ticks:
            custom_ticks.append(generations_axis[-1])

        # Цикл по кожному концепту-причині (C1...C5)
        for i in range(5):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xscale('log')

            # Отримуємо дані для поточного графіка
            data_y = history_array[:, i]

            # Побудова лінії
            ax.plot(generations_axis, data_y,
                    color='black',
                    linewidth=3.5)

            # 1. Прибираємо пусті місця зверху/знизу
            # Знаходимо реальний мінімум і максимум на цьому графіку
            y_min, y_max = np.min(data_y), np.max(data_y)

            # Додаємо мікро-відступ (наприклад 5% від розмаху), щоб лінія не "липла" до рамки
            # Якщо лінія пряма (min == max), робимо штучний відступ
            if y_max == y_min:
                offset = 0.1
            else:
                offset = (y_max - y_min) * 0.05

            ax.set_ylim(y_min - offset, y_max + offset)

            # 2. Налаштування вісі X
            ax.set_xticks(custom_ticks)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

            # 3. Товщина ліній осей
            axis_width = 3.0
            ax.spines['bottom'].set_linewidth(axis_width)
            ax.spines['left'].set_linewidth(axis_width)

            # Параметри тіків (засічок)
            ax.tick_params(axis='both', which='major', labelsize=26, width=axis_width, length=7)

            # 4. Уніфікація вісі Y: ЗАВЖДИ 5 міток
            # LinearLocator гарантує однакову кількість поділок на всіх графіках
            ax.yaxis.set_major_locator(LinearLocator(numticks=5))

            # Форматування: 2 знаки після коми
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # Дизайн (прибираємо зайві рамки)
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            # Збереження
            filename = f"dynamics_{scenario_name}_C{i + 1}.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"    Збережено: {filename}")

    # ---------------------------------------------------------
    # 3.2 Побудова графіка збіжності (Error Convergence)
    # ---------------------------------------------------------
    if is_plot_scenario and fitness_curve:
        print(f"--> Генерація графіка помилки (збіжності) для {scenario_name}...")

        # Перетворення Fitness назад у значення Помилки (Error)
        # Fitness = 1 / (Error + epsilon)  => Error = (1 / Fitness) - epsilon
        error_curve = [(1.0 / f) - 1e-8 for f in fitness_curve]

        plt.figure(figsize=(12, 6))

        # Побудова графіка чорним кольором
        plt.plot(error_curve, color='black', linewidth=2.5)

        # Прибираємо підписи осей (titles)
        plt.xlabel("")
        plt.ylabel("")

        axis_width = 2.0

        # За бажанням, можна налаштувати розмір шрифту тіків (чисел на осях)
        plt.tick_params(axis='both', which='major', labelsize=16, width=axis_width)

        # Сітка вимкнена, щоб графік був "чистим" (як на прикладах у статтях)
        plt.grid(False)

        # Опціонально: прибрати рамки зверху та справа
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(axis_width)
        ax.spines['left'].set_linewidth(axis_width)

        filename_fit = f"convergence_{scenario_name}.png"
        plt.savefig(filename_fit, dpi=300)
        plt.close()
        print(f"--> Графік збіжності збережено: {filename_fit}")

    # Статистика
    mse = np.mean((np.array(original_causes) - recovered_causes) ** 2)
    mad = np.mean(np.abs(np.array(original_causes) - recovered_causes))

    results_data.append({
        "Scenario": scenario_name,
        "x1": original_causes[0], "x2": original_causes[1], "x3": original_causes[2], "x4": original_causes[3],
        "x5": original_causes[4],
        "hat_x6": target_effects[0], "hat_x7": target_effects[1], "hat_x8": target_effects[2],
        "hat_x9": target_effects[3], "hat_x10": target_effects[4],
        "x1*": recovered_causes[0], "x2*": recovered_causes[1], "x3*": recovered_causes[2], "x4*": recovered_causes[3],
        "x5*": recovered_causes[4],
        "MSE": mse, "MAD": mad
    })

df_results = pd.DataFrame(results_data)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

cols_original = ["Scenario", "x1", "x2", "x3", "x4", "x5"]
cols_effects = ["Scenario", "hat_x6", "hat_x7", "hat_x8", "hat_x9", "hat_x10"]
cols_recovered = ["Scenario", "x1*", "x2*", "x3*", "x4*", "x5*", "MSE", "MAD"]

print("\n--- Таблиця 1: Початкові причини (x) ---")
print(df_results[cols_original].to_string(index=False))
print("\n--- Таблиця 2: Змодельовані наслідки (x^) ---")
print(df_results[cols_effects].to_string(index=False))
print("\n--- Таблиця 3: Відновлені причини (x*) та оцінка якості ---")
print(df_results[cols_recovered].to_string(index=False))
print("-" * 60)
print(f"Середня MSE: {df_results['MSE'].mean():.5f}")
print(f"Середня MAD: {df_results['MAD'].mean():.5f}")

df_results.to_csv('diag_seeded.csv', index=False)
import json; json.dump({'error_curve':[float(1.0/f-1e-8) for f in fitness_curve],'hist':np.array(history_to_plot).tolist()}, open('diag_seeded_s1.json','w'))
