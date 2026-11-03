import numpy as np
import pandas as pd
import pygad
import matplotlib.pyplot as plt

# ==========================================
# 0. НАЛАШТУВАННЯ ТИПІВ НОРМУВАННЯ (ТУМБЛЕР)
# ==========================================
NORM_RULES = {
    0: 'b',  # C1 (Вік): чим менше, тим краще
    1: 'a',  # C2 (Піковий ПД): чим більше, тим краще (резерв)
    2: 'a',  # C3 (Толерантність): чим більше, тим краще
    3: 'b',  # C4 (Приріст ПД/кг маси): менша "ціна" роботи - краще
    4: 'b',  # C5 (Приріст ПД/навантаження): менше - краще
    5: 'b',  # C6 (АТФ у крові): менше - краще (маркер пошкодження)
    6: 'b',  # C7 (АДФ): менше - краще
    7: 'a',  # C8 (АМФ): більше - краще (вазодилататор)
    8: 'a',  # C9 (Коеф. фосфорилювання): більше - краще
    9: 'a',  # C10 (МСК / VO2 max): більше - краще
    10: 'a',  # C11 (Приріст ПД субмакс): більше - краще
    11: 'b'  # C12 (Лактат/Піруват): менше - краще (немає гіпоксії)
}

FUZZY_C = 0.389

# ==========================================
# 1. ВХІДНІ ДАНІ ТА ПОПЕРЕДНЯ ОБРОБКА
# ==========================================
experimental_data = [
    [31, 324, 980, 2.8, 0.12, 50.07, 22.76, 8.05, 3.7, 34.2, 266, 19.3, 1],
    [36, 330, 900, 2.9, 0.14, 56.52, 24.33, 9.02, 4.1, 29.7, 242, 21.0, 1],
    [39, 260, 800, 2.3, 0.18, 51.73, 25.62, 8.53, 4.2, 28.5, 194, 23.8, 2],
    [42, 272, 867, 2.5, 0.28, 59.31, 28.44, 8.53, 4.0, 28.7, 198, 19.4, 2],
    [48, 287, 491, 2.2, 0.24, 52.77, 21.61, 8.53, 3.5, 25.3, 156, 20.5, 3],
    [53, 175, 507, 2.4, 0.25, 60.70, 26.14, 10.40, 3.9, 22.4, 172, 26.1, 3],
    [45, 247, 728, 2.0, 0.34, 62.06, 26.14, 5.55, 2.3, 26.5, 144, 22.9, 4],
    [52, 231, 768, 1.5, 0.36, 62.77, 23.01, 6.83, 2.5, 20.0, 158, 23.8, 4],
    [32, 151, 610, 1.3, 0.42, 54.49, 23.91, 5.55, 2.4, 19.8, 104, 25.7, 5],
    [45, 177, 542, 1.6, 0.48, 62.06, 26.14, 5.55, 2.3, 21.7, 120, 28.1, 5],
    [38, 128, 349, 1.4, 0.48, 67.03, 24.46, 5.20, 1.9, 13.9, 92, 30.2, 6],
    [38, 145, 304, 1.2, 0.56, 64.15, 25.62, 7.11, 2.6, 14.4, 74, 25.5, 6],
    [40, 327, 930, 2.2, 0.24, 59.31, 25.62, 7.56, 3.3, 34.5, 347, 18.9, 1],
    [38, 348, 952, 1.8, 0.20, 34.48, 20.79, 9.56, 5.7, 32.2, 352, 21.6, 1],
    [34, 307, 800, 1.9, 0.21, 57.90, 25.08, 6.83, 2.9, 30.1, 304, 19.3, 2],
    [48, 284, 738, 2.0, 0.26, 62.06, 25.08, 8.53, 3.4, 29.7, 339, 20.4, 2],
    [35, 174, 600, 1.7, 0.32, 55.18, 24.46, 8.56, 3.8, 27.2, 312, 22.0, 3],
    [49, 229, 515, 2.1, 0.30, 61.34, 22.20, 6.83, 2.4, 22.4, 300, 23.4, 3],
    [58, 265, 421, 2.0, 0.26, 60.07, 22.76, 4.08, 1.8, 17.7, 258, 23.8, 4],
    [49, 330, 650, 1.5, 0.25, 69.49, 25.08, 6.83, 2.5, 20.3, 244, 22.0, 4],
    [48, 187, 475, 1.4, 0.34, 60.39, 23.31, 5.55, 2.1, 21.4, 204, 22.7, 5],
    [42, 224, 400, 1.5, 0.39, 55.18, 21.05, 7.11, 2.7, 20.4, 215, 22.7, 5],
    [32, 195, 100, 1.2, 0.48, 60.70, 21.61, 7.52, 2.7, 22.6, 191, 25.9, 6],
    [51, 192, 292, 1.3, 0.45, 62.77, 23.70, 5.55, 1.6, 19.2, 188, 24.4, 6],
    [36, 347, 952, 2.9, 0.10, 62.40, 23.70, 12.50, 4.3, 35.7, 298, 19.6, 1],
    [48, 314, 902, 3.2, 0.14, 59.40, 24.20, 10.50, 4.2, 33.5, 287, 18.8, 1],
    [42, 352, 875, 3.2, 0.16, 52.30, 22.70, 9.50, 3.9, 38.2, 322, 19.0, 1],
    [40, 323, 1040, 2.7, 0.20, 59.60, 25.20, 8.80, 3.2, 30.4, 209, 18.2, 1],
    [41, 377, 988, 2.9, 0.09, 60.40, 24.30, 10.20, 3.4, 32.5, 275, 17.7, 1],
    [34, 309, 932, 3.2, 0.15, 60.80, 25.40, 9.40, 4.4, 31.5, 312, 18.5, 1],
    [52, 279, 1056, 2.7, 0.09, 59.90, 21.30, 8.80, 3.7, 33.4, 334, 18.7, 1],
    [44, 376, 895, 2.7, 0.18, 61.50, 23.60, 9.50, 3.6, 30.4, 312, 20.1, 2],
    [46, 304, 929, 2.6, 0.22, 58.20, 25.10, 10.70, 3.8, 32.5, 346, 19.2, 2],
    [46, 292, 904, 2.2, 0.24, 56.00, 27.90, 10.10, 4.0, 29.3, 290, 18.5, 2],
    [42, 276, 885, 2.4, 0.25, 61.40, 29.40, 11.20, 3.6, 27.8, 226, 20.8, 2],
    [31, 311, 930, 2.7, 0.19, 62.50, 23.80, 9.80, 2.9, 25.6, 249, 21.0, 2],
    [44, 335, 992, 2.4, 0.22, 61.60, 24.70, 9.90, 3.3, 24.6, 255, 20.3, 2],
    [47, 346, 873, 2.3, 0.18, 57.70, 22.50, 10.60, 3.7, 28.7, 267, 18.8, 2],
    [48, 288, 804, 2.4, 0.27, 60.00, 22.20, 11.50, 3.5, 20.9, 275, 19.5, 3],
    [50, 316, 875, 2.1, 0.31, 61.40, 24.00, 9.30, 2.8, 22.5, 302, 21.2, 3],
    [51, 292, 774, 2.0, 0.28, 62.50, 25.90, 8.80, 3.0, 26.7, 277, 22.7, 3],
    [54, 315, 766, 2.2, 0.22, 53.70, 26.20, 8.70, 2.7, 21.4, 265, 20.5, 3],
    [40, 300, 865, 2.1, 0.25, 59.40, 25.80, 9.30, 3.5, 21.9, 303, 21.4, 3],
    [36, 270, 777, 2.1, 0.28, 61.00, 26.10, 9.70, 4.1, 22.3, 316, 21.3, 3],
    [34, 275, 859, 2.3, 0.30, 62.50, 27.00, 9.60, 4.2, 24.0, 295, 22.5, 3],
    [52, 261, 776, 1.7, 0.36, 65.00, 22.50, 8.40, 2.7, 20.4, 204, 23.8, 4],
    [41, 258, 785, 1.5, 0.36, 62.70, 23.80, 7.60, 2.5, 19.8, 225, 24.0, 4],
    [53, 290, 845, 1.8, 0.39, 57.10, 24.00, 7.20, 2.5, 18.7, 268, 22.5, 4],
    [39, 203, 723, 2.0, 0.40, 58.50, 23.70, 6.20, 2.8, 17.1, 209, 24.7, 4],
    [45, 244, 802, 1.7, 0.35, 62.00, 25.30, 6.30, 3.0, 18.5, 212, 24.9, 4],
    [46, 233, 795, 1.9, 0.39, 57.90, 24.90, 5.20, 2.4, 17.4, 251, 23.5, 4],
    [54, 262, 805, 1.8, 0.38, 57.90, 24.50, 7.70, 2.2, 19.2, 244, 22.1, 4],
    [51, 245, 595, 1.3, 0.44, 64.20, 26.40, 5.60, 2.1, 16.5, 204, 24.7, 5],
    [40, 209, 772, 1.5, 0.45, 60.20, 27.80, 5.90, 2.4, 14.7, 195, 25.0, 5],
    [42, 198, 621, 1.4, 0.42, 58.80, 25.20, 6.10, 2.6, 12.2, 225, 24.5, 5],
    [44, 245, 523, 1.5, 0.39, 57.50, 23.30, 6.50, 2.2, 14.1, 207, 26.9, 5],
    [50, 237, 652, 1.6, 0.45, 63.70, 24.70, 6.40, 2.1, 11.9, 262, 24.2, 5],
    [56, 202, 744, 1.3, 0.45, 61.80, 25.70, 5.70, 2.4, 12.3, 226, 22.6, 5],
    [51, 247, 723, 1.2, 0.38, 62.50, 26.90, 5.60, 2.3, 10.4, 230, 25.8, 5],
    [48, 192, 516, 1.1, 0.52, 60.10, 22.70, 5.50, 2.0, 9.9, 200, 22.9, 6],
    [39, 188, 446, 1.2, 0.48, 59.00, 23.50, 5.20, 2.4, 9.5, 212, 26.7, 6],
    [49, 212, 406, 0.9, 0.56, 61.70, 26.00, 5.30, 1.9, 8.2, 225, 29.4, 6],
    [45, 247, 527, 0.7, 0.51, 62.60, 27.40, 5.10, 2.0, 7.4, 197, 28.5, 6],
    [44, 206, 448, 0.8, 0.55, 57.40, 22.10, 6.30, 2.1, 7.4, 188, 30.1, 6],
    [42, 228, 512, 1.0, 0.52, 53.90, 25.60, 5.40, 2.3, 7.8, 204, 29.5, 6]
]
raw_data = np.array(experimental_data)


def normalize_data(data, rules, c_param=FUZZY_C):
    norm_data = np.zeros_like(data, dtype=float)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    for j in range(data.shape[1]):
        xmin = min_vals[j]
        xmax = max_vals[j]

        if j < 12:
            rule = rules.get(j, 'a')

            # 1. Знаходимо u_i (відносне лінійне зміщення від 0 до 1)
            if xmax - xmin == 0:
                u = np.ones(data.shape[0])
            else:
                u = (data[:, j] - xmin) / (xmax - xmin)

            # 2. Застосовуємо Формулу (8) - дзвоноподібна функція Коші
            if rule == 'a':
                # Чим більше, тим краще (еквівалентно b_i = 1)
                norm_data[:, j] = 1.0 / (1.0 + ((u - 1.0) / c_param)**2)

            elif rule == 'b':
                # Чим менше, тим краще (еквівалентно b_i = 0)
                norm_data[:, j] = 1.0 / (1.0 + ((u - 0.0) / c_param)**2)

            elif rule == 'c':
                # Золота середина (еквівалентно b_i = 0.5)
                norm_data[:, j] = 1.0 / (1.0 + ((u - 0.5) / c_param)**2)

        else:
            # Діагноз (стовпець 13) не є концептом-симптомом,
            # тому залишаємо його у класичному лінійному вигляді
            if xmax - xmin == 0:
                norm_data[:, j] = 1.0
            else:
                norm_data[:, j] = (data[:, j] - xmin) / (xmax - xmin)

    return norm_data


def calculate_similarity_and_weights(A, weight_threshold=0.0):
    N, m = A.shape
    W = np.zeros((m, m))
    clns_matrix = np.zeros((m, m, N))
    signs = np.zeros((m, m))
    S = np.zeros((m, m))

    for j in range(m):
        for i in range(m):
            if i == j: continue
            d_direct = np.abs(A[:, i] - A[:, j])
            d_inverse = np.abs(A[:, i] - (1 - A[:, j]))
            if np.sum(d_direct) <= np.sum(d_inverse):
                clns_matrix[i, j, :] = d_direct
                signs[i, j] = 1
            else:
                clns_matrix[i, j, :] = d_inverse
                signs[i, j] = -1
            S[i, j] = signs[i, j] * (1 - np.mean(clns_matrix[i, j, :]))

    for j in range(m):
        INFL_j = [i for i in range(m) if i != j and abs(S[i, j]) > 0.7]
        if not INFL_j: continue
        clns_min = np.min(clns_matrix[INFL_j, j, :], axis=0)
        contribs = {}
        sum_contribs = 0.0
        for i in INFL_j:
            contrib_ij = 1 - np.mean(np.abs(clns_min - clns_matrix[i, j, :]))
            val = max(0, contrib_ij - 0.7)
            contribs[i] = val
            sum_contribs += val

        for i in INFL_j:
            if sum_contribs > 0:
                weight = signs[i, j] * (contribs[i] / sum_contribs)
                if abs(weight) >= weight_threshold:
                    W[i, j] = weight
                else:
                    W[i, j] = 0.0
    return W


def simulate_fcm_raw(A_initial, W, iterations=30):
    N, n = A_initial.shape
    A_pred = np.zeros((N, n))
    for p in range(N):
        A_k_minus_1 = np.zeros(n)
        A_k = A_initial[p, :].copy()
        for _ in range(iterations):
            delta_A = A_k - A_k_minus_1
            A_next = A_k + np.dot(delta_A, W)
            A_k_minus_1 = A_k.copy()
            A_k = A_next.copy()
        A_pred[p, :] = A_k
    return A_pred


def calculate_bounds(W, iterations=30):
    n = W.shape[0]
    A_min_input = np.zeros((1, n))
    A_max_input = np.zeros((1, n))
    for i in range(n - 1):
        if W[i, -1] > 0:
            A_max_input[0, i] = 1.0
        elif W[i, -1] < 0:
            A_min_input[0, i] = 1.0

    A_min_steady = simulate_fcm_raw(A_min_input, W, iterations)
    A_max_steady = simulate_fcm_raw(A_max_input, W, iterations)
    return A_min_steady[0, -1], A_max_steady[0, -1]


# ==========================================
# 2. ПІДГОТОВКА ДАНИХ ТА СТВОРЕННЯ ХРОМОСОМИ [ W | T ]
# ==========================================
A_norm_all = normalize_data(raw_data, NORM_RULES)

real_output_all = A_norm_all[:, -1]
A_input_all = A_norm_all.copy()
A_input_all[:, -1] = 0.0
real_diag_all = raw_data[:, -1].astype(int)

np.random.seed(42)
indices = np.random.permutation(len(raw_data))

train_idx = indices[:]
test_idx = indices[:]

A_norm_train = A_norm_all[train_idx]
A_input_train = A_input_all[train_idx]
real_output_train = real_output_all[train_idx]
real_diag_train = real_diag_all[train_idx]

A_input_test = A_input_all[test_idx]
real_output_test = real_output_all[test_idx]
real_diag_test = real_diag_all[test_idx]

W_initial = calculate_similarity_and_weights(A_norm_train, weight_threshold=0.09)
W_initial[-1, :] = 0.0
W_initial[:, 0] = 0.0

# 2.1 Формування об'єднаної структури генів: Ваги + Пороги
gene_indices = []
gene_space = []
initial_genes = []

n_concepts = W_initial.shape[0]
for i in range(n_concepts - 1):
    for j in range(n_concepts):
        if W_initial[i, j] != 0:
            gene_indices.append((i, j))
            w = W_initial[i, j]
            initial_genes.append(w)
            if w > 0:
                gene_space.append({'low': max(0.02, w - 0.3), 'high': min(0.98, w + 0.3)})
            else:
                gene_space.append({'low': max(-0.98, w - 0.3), 'high': min(-0.02, w + 0.3)})

num_weights = len(gene_indices)

# Додаємо пороги до хромосоми
manual_thresholds = [0.20, 0.35, 0.50, 0.65, 0.80]
initial_genes.extend(manual_thresholds)
for _ in range(5):
    gene_space.append({'low': 0.05, 'high': 0.95})

print('Матриця ваг до оптимізації (побудована на 50 зразках)')
labels = [f"C{i + 1}" for i in range(n_concepts)]
df_initial = pd.DataFrame(W_initial, index=labels, columns=labels)
print(df_initial.to_string(float_format=lambda x: f"{x:.3f}"))


# ==========================================
# 3. НАЛАШТУВАННЯ ЄДИНОГО ГІБРИДНОГО ГЕНЕТИЧНОГО АЛГОРИТМУ
# ==========================================
def predict_and_evaluate(W_temp, A_inp, r_out):
    min_A, max_A = calculate_bounds(W_temp)
    A_steady = simulate_fcm_raw(A_inp, W_temp)
    pred_raw = A_steady[:, -1]

    if max_A - min_A == 0:
        pred_norm = np.clip(pred_raw, 0, 1)
    else:
        pred_norm = np.clip((pred_raw - min_A) / (max_A - min_A), 0, 1)

    ssd = np.sum((pred_norm - r_out) ** 2)
    return ssd, pred_norm


def fitness_func(ga_instance, solution, solution_idx):
    # 1. Розпакування ваг
    W_temp = W_initial.copy()
    for idx, (i, j) in enumerate(gene_indices):
        W_temp[i, j] = solution[idx]

    # 2. Розпакування та сортування порогів
    raw_thresholds = solution[num_weights:]
    thresholds = sorted(raw_thresholds)

    # 3. Моделювання
    ssd, pred_norm = predict_and_evaluate(W_temp, A_input_train, real_output_train)

    def classify(val):
        if val < thresholds[0]:
            return 1
        elif val < thresholds[1]:
            return 2
        elif val < thresholds[2]:
            return 3
        elif val < thresholds[3]:
            return 4
        elif val < thresholds[4]:
            return 5
        else:
            return 6

    current_predictions = np.array([classify(v) for v in pred_norm])
    matches = np.sum(current_predictions == real_diag_train)
    penalty = np.sum(np.abs(current_predictions - real_diag_train))

    # 4. Гібридна функція (максимізація)
    fitness = (matches * 10) - penalty + (1.0 / (ssd + 1e-4))
    return fitness


true_error_history = []


def on_generation(ga_instance):
    solution, _, _ = ga_instance.best_solution()

    W_temp = W_initial.copy()
    for idx, (i, j) in enumerate(gene_indices):
        W_temp[i, j] = solution[idx]

    ssd, _ = predict_and_evaluate(W_temp, A_input_train, real_output_train)
    mse = ssd / len(train_idx)

    if true_error_history:
        mse = min(mse, true_error_history[-1])

    true_error_history.append(mse)


pop_size = 50  # Збільшена популяція через об'єднання параметрів
initial_population = [initial_genes]
for _ in range(pop_size - 1):
    random_genes = []
    # Випадкові ваги
    for i in range(num_weights):
        space = gene_space[i]
        random_genes.append(np.random.uniform(space['low'], space['high']))
    # Випадкові пороги
    rand_t = sorted(np.random.uniform(0.05, 0.95, 5))
    random_genes.extend(rand_t)
    initial_population.append(random_genes)

ga_instance = pygad.GA(
    num_generations=400,
    num_parents_mating=15,
    fitness_func=fitness_func,
    sol_per_pop=pop_size,
    num_genes=len(initial_genes),
    gene_space=gene_space,
    initial_population=initial_population,
    parent_selection_type="sss",
    keep_parents=5,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=15,
    on_generation=on_generation,
    random_seed=42
)

# ==========================================
# 4. ВИКОНАННЯ ТА ГЕНЕРАЦІЯ ГРАФІКА ПРИСТОСУВАННЯ
# ==========================================
print("\nЗапуск Комплексного Генетичного Алгоритму...")
ga_instance.run()

best_solution, _, _ = ga_instance.best_solution()

# Відновлюємо найкращі ваги
W_optimized = W_initial.copy()
for idx, (i, j) in enumerate(gene_indices):
    W_optimized[i, j] = best_solution[idx]

# Відновлюємо найкращі пороги
best_thresholds = sorted(best_solution[num_weights:])

print('\nМатриця ваг після оптимізації')
df_optimized = pd.DataFrame(W_optimized, index=labels, columns=labels)
print(df_optimized.to_string(float_format=lambda x: f"{x:.3f}"))

print(f"\nОптимальні пороги знайдено: {np.round(best_thresholds, 4)}")

# ==========================================
# ПОБУДОВА ГРАФІКА МІНІМІЗАЦІЇ ПОХИБКИ (ERROR)
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(range(len(true_error_history)), true_error_history, color='black', linewidth=2)
plt.title("Динаміка Комплексної Оптимізації (Мінімізація Похибки)")
plt.ylabel("Середньоквадратична похибка (MSE)")
plt.xlabel("Ітерація (Generation)")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('optimization_dynamics.png', dpi=500)
print("\nГрафік динаміки оптимізації (Спадаючий) збережено як 'optimization_dynamics.png'")


def final_classify(val):
    if val < best_thresholds[0]:
        return 1
    elif val < best_thresholds[1]:
        return 2
    elif val < best_thresholds[2]:
        return 3
    elif val < best_thresholds[3]:
        return 4
    elif val < best_thresholds[4]:
        return 5
    else:
        return 6


def manual_classify(val):
    if val < manual_thresholds[0]:
        return 1
    elif val < manual_thresholds[1]:
        return 2
    elif val < manual_thresholds[2]:
        return 3
    elif val < manual_thresholds[3]:
        return 4
    elif val < manual_thresholds[4]:
        return 5
    else:
        return 6


# ==========================================
# 6. ФІНАЛЬНА ПЕРЕВІРКА ТОЧНОСТІ ТА ТАБЛИЦІ
# ==========================================
def print_evaluation(dataset_name, W_matrix, classify_func, A_inp, r_out, r_diag, original_indices):
    _, pred_norm = predict_and_evaluate(W_matrix, A_inp, r_out)
    final_preds = np.array([classify_func(v) for v in pred_norm])
    MAD = np.mean(np.abs(pred_norm - r_out))
    MSE = np.mean((pred_norm - r_out) ** 2)
    matches = np.sum(final_preds == r_diag)

    # Гарантоване перетворення в int
    differences = np.abs(r_diag - final_preds).astype(int)

    df_comp = pd.DataFrame({
        "Реальний діагноз": r_diag.astype(int),
        "Прогноз моделі": final_preds,
        "Похибка": ["*" * diff for diff in differences]
    }, index=original_indices)
    df_comp.index.name = 'Рядок №'

    print(f"\n=======================================================")
    print(f"       {dataset_name} ({len(r_diag)} зразків)")
    print(f"=======================================================\n")
    print("=== ПОРІВНЯННЯ ПОХИБОК (БЕЗПЕРЕРВНИХ ДАНИХ) ===")
    print(f"MAD = {MAD:.4f}, MSE = {MSE:.4f}")

    print("\n=== ДИСКРЕТНА ТОЧНІСТЬ (Точні збіги класів) ===")
    print(f"{matches} з {len(r_diag)} ({(matches / len(r_diag)) * 100:.1f}%)")

    print("\n=== ДЕТАЛЬНА ТАБЛИЦЯ ПРОГНОЗІВ ===")
    print(df_comp.to_string())


print("\n\n#######################################################")
print("      РЕЗУЛЬТАТИ ДО НАЛАШТУВАННЯ (ПОЧАТКОВІ ВАГИ ТА ПОРОГИ)")
print("#######################################################")

print_evaluation("ТЕСТОВА ВИБІРКА (ДО ГА)", W_initial, manual_classify, A_input_test, real_output_test, real_diag_test,
                 test_idx)
print_evaluation("УСЯ ВИБІРКА (ДО ГА)", W_initial, manual_classify, A_input_all, real_output_all, real_diag_all,
                 np.arange(len(raw_data)))

print("\n\n#######################################################")
print("      РЕЗУЛЬТАТИ ПІСЛЯ КОМПЛЕКСНОЇ ОПТИМІЗАЦІЇ [W | T]")
print("#######################################################")

print_evaluation("ТЕСТОВА ВИБІРКА (ПІСЛЯ ГА)", W_optimized, final_classify, A_input_test, real_output_test,
                 real_diag_test, test_idx)
print_evaluation("УСЯ ВИБІРКА (ПІСЛЯ ГА)", W_optimized, final_classify, A_input_all, real_output_all, real_diag_all,
                 np.arange(len(raw_data)))

# =======================================================
# 7. ГЕНЕРАЦІЯ ТАБЛИЦЬ 3, 4, 5, 6 (CSV)
# =======================================================
print("\n\n-------------------------------------------------------")
print(" ГЕНЕРАЦІЯ ТАБЛИЦЬ 3, 4, 5, 6 (CSV)")
print("-------------------------------------------------------")


def get_interval(diag_class, thresh):
    if diag_class == 1:
        return f"[0.000; {thresh[0]:.3f})"
    elif diag_class == 2:
        return f"[{thresh[0]:.3f}; {thresh[1]:.3f})"
    elif diag_class == 3:
        return f"[{thresh[1]:.3f}; {thresh[2]:.3f})"
    elif diag_class == 4:
        return f"[{thresh[2]:.3f}; {thresh[3]:.3f})"
    elif diag_class == 5:
        return f"[{thresh[3]:.3f}; {thresh[4]:.3f})"
    else:
        return f"[{thresh[4]:.3f}; 1.000]"


df_raw = pd.DataFrame(raw_data, columns=[f"x{i + 1}" for i in range(12)] + ["Diagnosis"])
df_raw.index += 1
df_raw.to_csv('Table_3_Clinical_Data.csv', index_label="Patient")
print("-> Таблицю 3 збережено у 'Table_3_Clinical_Data.csv'")

df_norm_out = pd.DataFrame()
for i in range(12):
    df_norm_out[f"A{i + 1}"] = [f"{val:.3f}" for val in A_norm_all[:, i]]
df_norm_out["A13"] = [get_interval(int(d), best_thresholds) for d in real_diag_all]
df_norm_out.index += 1
df_norm_out.to_csv('Table_4_Normalized_Data.csv', index_label="Patient")
print("-> Таблицю 4 збережено у 'Table_4_Normalized_Data.csv'")

connections = []
for i in range(n_concepts):
    for j in range(n_concepts):
        w_in = W_initial[i, j]
        w_opt = W_optimized[i, j]
        if abs(w_in) > 0.001 or abs(w_opt) > 0.001:
            connections.append({
                "Зв'язок": f"w_{i + 1},{j + 1}",
                "До налаштування": round(w_in, 3),
                "Після налаштування": round(w_opt, 3)
            })
df_table5 = pd.DataFrame(connections)
df_table5.to_csv('Table_5_Weights.csv', index=False)
print("-> Таблицю 5 збережено у 'Table_5_Weights.csv'")

_, pred_norm_all_before = predict_and_evaluate(W_initial, A_input_all, real_output_all)
diag_before = [manual_classify(v) for v in pred_norm_all_before]
_, pred_norm_all_after = predict_and_evaluate(W_optimized, A_input_all, real_output_all)
diag_after = [final_classify(v) for v in pred_norm_all_after]

table6_data = []
for i in range(len(raw_data)):
    real_d = int(real_diag_all[i])
    table6_data.append({
        "Номер": i + 1,
        "Реальний діагноз (d_i)": f"d_{real_d}",
        "Інтервал (A_13)": get_interval(real_d, best_thresholds),
        "A_13 До налаштування": f"{pred_norm_all_before[i]:.3f}",
        "d_i До налаштування": f"d_{diag_before[i]}",
        "A_13 Після налаштування": f"{pred_norm_all_after[i]:.3f}",
        "d_i Після налаштування": f"d_{diag_after[i]}"
    })
df_table6 = pd.DataFrame(table6_data)
df_table6.to_csv('Table_6_Comparison.csv', index=False)
print("-> Таблицю 6 збережено у 'Table_6_Comparison.csv'")
print("-------------------------------------------------------")

# =======================================================
# 8. ГЕНЕРАЦІЯ ТАБЛИЦІ 7 (ВИВІД У КОНСОЛЬ)
# =======================================================
print("\n\n---------------------------------------------------------------------------------------------------------")
print(" Таблиця 7. Результати діагностики за допомогою НКК (до / після налаштування)")
print("---------------------------------------------------------------------------------------------------------")

# Ініціалізація матриць 6x6 для підрахунку кількості діагнозів
matrix_before = np.zeros((6, 6), dtype=int)
matrix_after = np.zeros((6, 6), dtype=int)
real_counts = np.zeros(6, dtype=int)

# Підрахунок значень для кожної клітинки
for i in range(len(real_diag_all)):
    r = int(real_diag_all[i]) - 1  # Реальний діагноз (індекс 0-5)
    p_b = int(diag_before[i]) - 1  # Прогноз до налаштування (індекс 0-5)
    p_a = int(diag_after[i]) - 1  # Прогноз після налаштування (індекс 0-5)

    real_counts[r] += 1
    matrix_before[r, p_b] += 1
    matrix_after[r, p_a] += 1

# Формування та вивід таблиці
header = f"| {'Реальні':^9} | {'К-ть':^4} | {'d1':^9} | {'d2':^9} | {'d3':^9} | {'d4':^9} | {'d5':^9} | {'d6':^9} |"
separator = "-" * len(header)

print(separator)
print(f"| {' ':^9} | {' ':^4} | {'Обчислені діагнози (до / після)':^65} |")
print(header)
print(separator)

for i in range(6):
    row = f"| d{i + 1:<8} | {real_counts[i]:<4} |"
    for j in range(6):
        # Формат клітинки: "до/після"
        cell = f"{matrix_before[i, j]} / {matrix_after[i, j]}"
        row += f" {cell:^9} |"
    print(row)

print(separator)
print(f"| {'СУМА':<9} | {np.sum(real_counts):<4} |{' ':>68}|")
print(separator)

import itertools

# =======================================================
# 9. ОБРАХУНОК РАНГІВ (ІНДЕКСІВ ВАЖЛИВОСТІ) КОНЦЕПТІВ
# =======================================================
print("\n\n---------------------------------------------------------------------------------------------------------")
print(" Обрахунок рангів (індексів важливості) концептів")
print("---------------------------------------------------------------------------------------------------------")

n_nodes = W_optimized.shape[0]

# Крок 1. Знаходимо найменше (A_min) та найбільше (A_max) можливе значення вихідного концепту
# Формуємо всі можливі бінарні комбінації вхідних концептів (2^12 = 4096 комбінацій)
print("Пошук A_min та A_max перебором усіх можливих комбінацій вхідних концептів...")
all_combinations = list(itertools.product([0, 1], repeat=n_nodes - 1))

min_A_raw = float('inf')
max_A_raw = float('-inf')
min_combo = None
max_combo = None

for combo in all_combinations:
    A_init = np.zeros((1, n_nodes))
    A_init[0, :-1] = combo
    # Симулюємо НКК до стабільного стану
    A_steady = simulate_fcm_raw(A_init, W_optimized)
    out_val = A_steady[0, -1]

    if out_val < min_A_raw:
        min_A_raw = out_val
        min_combo = combo
    if out_val > max_A_raw:
        max_A_raw = out_val
        max_combo = combo

print(f"Знайдено A_min (нижнє): {min_A_raw:.3f} при комбінації {min_combo}")
print(f"Знайдено A_max (верхнє): {max_A_raw:.3f} при комбінації {max_combo}")

# Крок 2. Ранжування кожного вхідного концепту окремо
importance_indices = []

print("\nТаблиця рангів:")
header_ranks = f"| {'Концепт':^9} | {'Початковий вектор (входи)':^35} | {'I(C_i) (нормалізоване)':^22} |"
separator_ranks = "-" * len(header_ranks)
print(separator_ranks)
print(header_ranks)
print(separator_ranks)

for i in range(n_nodes - 1):
    # Проставляємо тільки поточний концепт як 1, решта 0
    A_init = np.zeros((1, n_nodes))
    A_init[0, i] = 1.0

    # Симуляція
    A_steady = simulate_fcm_raw(A_init, W_optimized)
    raw_out = A_steady[0, -1]

    # Нормалізація результату за формулою 11
    if max_A_raw - min_A_raw == 0:
        norm_out = 0.0
    else:
        norm_out = (raw_out - min_A_raw) / (max_A_raw - min_A_raw)

    importance_indices.append(norm_out)

    # Форматування для виводу у консоль
    combo_str = " ".join([str(int(x)) for x in A_init[0, :-1]])
    print(f"| C{i + 1:<8} | {combo_str:^35} | {norm_out:^22.3f} |")

print(separator_ranks)

# Крок 3. Побудова графіка
plt.figure(figsize=(10, 5))
x_labels = [f"$C_{{{i + 1}}}$" for i in range(n_nodes - 1)]

# Створення стовпчикової діаграми (біла заливка, чорна рамка)
bars = plt.bar(x_labels, importance_indices, color='white', edgecolor='black', linewidth=1.5)

plt.title("$I(C_i)$", fontsize=16, loc='left', pad=15)

# Налаштування осей
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tick_params(axis='both', which='major', labelsize=14)

# Додавання мітки осі X як C_i в кінці графіка
plt.text(len(x_labels), -max(importance_indices) * 0.05, "$C_i$", fontsize=16, va='top', ha='left')

plt.tight_layout()
plt.savefig('concept_ranking.png', dpi=500)
print("\nГрафік рангів збережено як 'concept_ranking.png'")
