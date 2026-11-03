# Оригінал: "FORECASTING AND DIAGNOSTICS IN GENERALIZED DYNAMIC SYSTEMS BASED ON FUZZY COGNITIVE MAPS.py"
import numpy as np
import matplotlib.pyplot as plt

# 1. Матриця впливів W (експертна)
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


def simulate_and_plot(x_u):
    idx_causes = [0, 1, 2, 3, 4]
    idx_effects = [5, 6, 7, 8, 9]

    x_curr = np.zeros(10)
    x_curr[idx_causes] = x_u
    x_prev = np.zeros(10)

    history = [x_curr.copy()]

    # Обмеження до 15 ітерацій
    for _ in range(15):
        delta = x_curr - x_prev
        influence = np.dot(delta, W)
        x_next = x_curr + influence
        history.append(x_next.copy())

        if np.max(np.abs(x_next - x_curr)) < 0.0001:
            x_curr = x_next
            break
        x_prev = x_curr
        x_curr = x_next

    history = np.array(history)

    plt.figure(figsize=(12, 7))

    # Легка пунктирна сітка на фоні
    plt.grid(True, linestyle='--', alpha=0.5, color='silver', zorder=0)

    # Малюємо всі лінії суцільними та однаковими
    for i in range(10):
        plt.plot(history[:, i],
                 color='#1a1a1a',
                 linestyle='-',
                 linewidth=2.5,
                 zorder=2)

    # --- АЛГОРИТМ ДИНАМІЧНОГО РОЗСУНЕННЯ КОНЦЕПТІВПраворуч ---
    final_vals = history[-1].copy()
    sorted_idx = np.argsort(final_vals)
    adjusted_y = final_vals.copy()

    # Збільшена відстань між підписами, щоб уникнути зліплення
    min_text_dist = 0.075

    for idx in range(1, len(sorted_idx)):
        curr = sorted_idx[idx]
        prev = sorted_idx[idx - 1]
        if adjusted_y[curr] - adjusted_y[prev] < min_text_dist:
            adjusted_y[curr] = adjusted_y[prev] + min_text_dist

    # Виведення підписів концептів (Великий шрифт 28)
    for i in range(10):
        plt.text(15.3, adjusted_y[i], f'$C_{{{i + 1}}}$',
                 va='center',
                 ha='left',
                 fontsize=28,
                 color='black')

    # Обмежуємо осі як у першому варіанті
    plt.xlim(0, 16.8)
    plt.ylim(-0.5, 1.05)

    # Великі засічки на осях (Шрифт 26)
    plt.xticks(range(0, 16, 2), fontsize=26)
    plt.yticks(fontsize=26)

    # Текстові підписи осей видалено згідно з інструкцією

    # ВІДНОВЛЕННЯ ОСЕЙ: Класичний академічний куточок (Ліва та Нижня осі включені)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)  # Повернули вісь Y ліворуч
    ax.spines['bottom'].set_visible(True)  # Повернули вісь X знизу
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig("stabilization_final.png", dpi=300, bbox_inches='tight')
    plt.show()


# Вхідні дані
x_input = [0.6, -0.2, -0.4, 0.2, 0.4]
simulate_and_plot(x_input)
