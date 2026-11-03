# Оригінал: "Метод Сааті main.py"
import numpy as np
import pandas as pd

# 1. Вхідна матриця (еталон з формули 13)
original_matrix = np.array([
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
concepts = [f"C{i + 1}" for i in range(10)]


def extract_influences(matrix):
    pos_links = []
    neg_links = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            weight = matrix[i, j]
            if weight > 0:
                pos_links.append({'source': i, 'target': j, 'value': weight, 'id': f"w{i + 1},{j + 1}"})
            elif weight < 0:
                neg_links.append({'source': i, 'target': j, 'value': weight, 'id': f"w{i + 1},{j + 1}"})
    return pos_links, neg_links


def get_saaty_linguistic(rank):
    """Повертає лінгвістичну оцінку для шкали Сааті."""
    rank = int(rank)
    if rank == 1: return "Такий самий (Equal)"
    if rank == 2: return "Між 1 і 3"
    if rank == 3: return "Трохи сильніший (Moderate)"
    if rank == 4: return "Між 3 і 5"
    if rank == 5: return "Сильніший (Strong)"
    if rank == 6: return "Між 5 і 7"
    if rank == 7: return "Значно сильніший (Very Strong)"
    if rank == 8: return "Між 7 і 9"
    if rank >= 9: return "Абсолютно сильніший (Extreme)"
    return str(rank)


def apply_least_influence_method_detailed(links_data, group_name):
    if not links_data: return pd.DataFrame(), None

    # 1. Знаходимо найменший вплив
    min_link = min(links_data, key=lambda x: abs(x['value']))
    w_min_abs = abs(min_link['value'])
    min_id = min_link['id']

    print(f"\n=== {group_name} ВПЛИВИ ===")
    print(f"Найменший вплив (еталон): {min_id} (вага {min_link['value']})")

    # 2. Розрахунок рангів Сааті (округлення до 1..9)
    saaty_data = []
    ratios_sum = 0

    for link in links_data:
        true_ratio = abs(link['value']) / w_min_abs
        # Округлення до найближчого цілого в межах [1, 9]
        saaty_rank = int(round(true_ratio))
        if saaty_rank < 1: saaty_rank = 1
        if saaty_rank > 9: saaty_rank = 9

        ratios_sum += saaty_rank

        saaty_data.append({
            'Link_ID': link['id'],
            'Source_Target': f"C{link['source'] + 1}->C{link['target'] + 1}",
            'True_Weight': link['value'],
            'True_Ratio': true_ratio,
            'Saaty_Rank': saaty_rank,
            'Linguistic': get_saaty_linguistic(saaty_rank),
            'Ratio_Str': f"r_{link['id'][1:]} / r_{min_id[1:]} = {saaty_rank}"
        })

    df_saaty = pd.DataFrame(saaty_data)

    # Вивід Таблиці аналог Табл. 5
    print(f"\nТаблиця порівнянь ({group_name}):")
    print(df_saaty[['Link_ID', 'Linguistic', 'Ratio_Str']].to_string(index=False))

    # 3. Розрахунок w_min за формулою (3)
    # w_min = 1 / (sum of ranks)
    w_min_calc = 1.0 / ratios_sum

    print(f"\nРозрахунок сили найменшого впливу (Формула 3):")
    formula_str = " + ".join([str(r) for r in df_saaty['Saaty_Rank'].tolist()])
    print(f"{min_id} = 1 / ({formula_str}) = 1 / {ratios_sum} ≈ {w_min_calc:.2f}")

    # 4. Розрахунок інших ваг за формулою (2)
    # w_i = w_min * rank_i
    calculated_results = []
    raw_weights_abs = []  # Для пошуку максимуму для нормалізації

    print(f"\nРозрахунок решти ваг (Формула 2) (до нормалізації):")
    for idx, row in df_saaty.iterrows():
        w_calc_abs = w_min_calc * row['Saaty_Rank']
        raw_weights_abs.append(w_calc_abs)

        sign = 1 if row['True_Weight'] > 0 else -1
        w_calc_signed = w_calc_abs * sign

        print(f"{row['Link_ID']} = {w_min_calc:.2f} * {row['Saaty_Rank']} = {w_calc_signed:.2f}")

        calculated_results.append({
            'Link_ID': row['Link_ID'],
            'Calculated_Raw': w_calc_signed,
            'Abs_Raw': w_calc_abs,
            'True_Weight': row['True_Weight']
        })

    # 5. Нормалізація
    max_w = max(raw_weights_abs)
    print(f"\nНормалізація (ділення на max = {max_w:.2f}):")

    final_data = []
    for item in calculated_results:
        w_norm = item['Calculated_Raw'] / max_w
        final_data.append({
            'Link': item['Link_ID'],
            'Calculated (Norm)': w_norm,
            'Original (Input)': item['True_Weight'],
            'Saaty Rank': df_saaty.loc[df_saaty['Link_ID'] == item['Link_ID'], 'Saaty_Rank'].values[0]
        })

    return pd.DataFrame(final_data), final_data


# Виконання
pos_links, neg_links = extract_influences(original_matrix)

df_res_pos, list_res_pos = apply_least_influence_method_detailed(pos_links, "ПОЗИТИВНІ")
print("\n" + "=" * 50 + "\n")
df_res_neg, list_res_neg = apply_least_influence_method_detailed(neg_links, "НЕГАТИВНІ")

# Зведена таблиця результатів
print("\n\n=== ПІДСУМКОВА ТАБЛИЦЯ ВАГ ===")
all_results = pd.concat([df_res_pos, df_res_neg])
print(all_results.to_string(index=False))

# Відновлення матриці
reconstructed_matrix = np.zeros((10, 10))
for item in list_res_pos + list_res_neg:
    # item['Link'] is like "w1,6" -> parse indices
    indices = item['Link'][1:].split(',')
    r, c = int(indices[0]) - 1, int(indices[1]) - 1
    reconstructed_matrix[r, c] = item['Calculated (Norm)']

print("\n=== ВІДНОВЛЕНА МАТРИЦЯ (Таблиця 13) ===")
df_matrix = pd.DataFrame(reconstructed_matrix, index=concepts, columns=concepts)
print(df_matrix.round(3).to_string())

def get_interval_for_weight(w):
    variation = 0.2

    if w > 0:
        lower = max(0.02, w - variation)
        upper = min(0.98, w + variation)
        return round(lower, 2), round(upper, 2)
    elif w < 0:
        lower = max(-0.98, w - variation)
        upper = min(-0.02, w + variation)
        return round(lower, 2), round(upper, 2)
    else:
        return 0.0, 0.0


print("\n" + "=" * 20 + " КОД ДЛЯ КОПІЮВАННЯ В СКРИПТ ТЮНІНГУ " + "=" * 20)
print("FCM_STRUCTURE = [")
for i in range(10):
    for j in range(10):
        w = reconstructed_matrix[i, j]
        if w != 0:
            min_val, max_val = get_interval_for_weight(w)
            # Формат: (Source_Index, Target_Index, Initial_Value, Min, Max)
            print(f"    {{'src': {i}, 'tgt': {j}, 'init': {w:.4f}, 'min': {min_val}, 'max': {max_val}}},")
print("]")
print("=" * 60)
