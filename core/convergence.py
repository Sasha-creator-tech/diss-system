"""Перевірка збіжності НКК з різними правилами виведення та пороговими (активаційними) функціями.

Розширення системи (не змінює формул статей дисертації). Джерело правил і функцій — огляд
G. Nápoles et al., «A Review on Fuzzy Cognitive Mapping: Recent Advances and Algorithms»,
Big Data Cogn. Comput. 2026, 10, 22 (далі «огляд»):

  правила виведення (рядок-вектор A, W[i, j] — вплив Ci → Cj, тож A·W — сума вхідних впливів):
    native    базова рекурентна формула статей дисертації (EMP-PAVLOV (9), код: fcm.simulate)
              A^{k+1} = A^k + (A^k − A^{k−1})·W,  A^1 = A^0 + A^0·W;   за потреби f(·) після кроку
    kosko     огляд (1)–(2): A^t = f(A^{t−1}·W)
    extended  огляд (3)–(4): A^t = f(k1·A^{t−1}·W + k2·A^{t−1}),  0 < k1, k2 ≤ 1
    selfloop  огляд (5):     A^t = f(A^{t−1}·W + w_ii·A^{t−1})  (однаковий самовплив w_ii, діагональ W = 0)
    quasi     огляд (6)–(7): A^t = φ·f(A^{t−1}·W) + (1 − φ)·A^0,  0 ≤ φ ≤ 1

  функції f — огляд, табл. 1: двозначна, ступінчаста (поріг v), тризначна, tanh, сигмоїда (λ, h),
  масштабувальна X/||X||₂ (на весь вектор), експоненційна нормована sgn(x)(|x|/Σ_j|w_ji|)^β;
  «none» — без функції (як у статтях дисертації, де замість f — нормування виходу (11)).

Стани динаміки (огляд, розд. 3): нерухома точка, граничний цикл, відсутність збіжності за T кроків
(хаос або повільна збіжність), розбіжність (значення > blowup).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------- функції активації (огляд, табл. 1)

UNIT, SYM = "unit", "sym"            # область значень функції: [0, 1] або [−1, 1]


@dataclass(frozen=True)
class Activation:
    key: str
    label: str
    formula: str
    image: str | None                 # UNIT / SYM / None (без обмеження)
    params: tuple[str, ...] = ()


ACTIVATIONS: dict[str, Activation] = {a.key: a for a in [
    Activation("none", "Без функції", "f(x) = x", None),
    Activation("bivalent", "Двозначна", "f(x) = 1, якщо x > 0; інакше 0", UNIT),
    Activation("step", "Ступінчаста", "f(x) = 1, якщо x > v; інакше 0", UNIT, ("v",)),
    Activation("trivalent", "Тризначна", "f(x) = sgn(x) ∈ {−1, 0, 1}", SYM),
    Activation("tanh", "Гіперболічний тангенс", "f(x) = tanh(λx)", SYM, ("lam_tanh",)),
    Activation("sigmoid", "Сигмоїда", "f(x) = 1 / (1 + e^(−λ(x − h)))", UNIT, ("lam", "h")),
    Activation("rescaled", "Масштабувальна", "f(X) = X / ||X||₂ (0, якщо X = 0)", SYM),
    Activation("expnorm", "Експоненційна нормована", "f(x) = sgn(x)·(|x| / Σⱼ|wⱼᵢ|)^β", SYM, ("beta",)),
]}


def activate(key: str, X: np.ndarray, W: np.ndarray, p: dict) -> np.ndarray:
    """f(X) для матриці станів X (N, n), рядок — один вектор концептів (огляд, табл. 1)."""
    if key == "none":
        return X
    if key == "bivalent":
        return (X > 0).astype(float)
    if key == "step":
        return (X > p["v"]).astype(float)
    if key == "trivalent":
        return np.sign(X)
    if key == "tanh":
        return np.tanh(p["lam_tanh"] * X)
    if key == "sigmoid":
        z = np.clip(-p["lam"] * (X - p["h"]), -700, 700)
        return 1.0 / (1.0 + np.exp(z))
    if key == "rescaled":
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        return np.divide(X, nrm, out=np.zeros_like(X), where=nrm > 0)
    if key == "expnorm":
        s = np.sum(np.abs(W), axis=0)                 # Σ_j |w_ji| — сума модулів вхідних ваг концепту i
        s = np.where(s > 0, s, 1.0)
        return np.sign(X) * (np.abs(X) / s) ** p["beta"]
    raise KeyError(key)


def lipschitz(key: str, p: dict) -> float | None:
    """Стала Ліпшиця f (для достатньої умови стиску); None — функція розривна або векторна."""
    return {"none": 1.0, "tanh": abs(p["lam_tanh"]), "sigmoid": abs(p["lam"]) / 4.0}.get(key)


# ---------------------------------------------------------------- правила виведення (огляд, (1)–(7))

@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    formula: str
    params: tuple[str, ...] = ()


RULES: dict[str, Rule] = {r.key: r for r in [
    Rule("native", "Базова (як у моделі)", "A⁽ᵏ⁺¹⁾ = f(A⁽ᵏ⁾ + (A⁽ᵏ⁾ − A⁽ᵏ⁻¹⁾)·W)"),
    Rule("kosko", "Класична (1)", "A⁽ᵗ⁾ = f(A⁽ᵗ⁻¹⁾·W)"),
    Rule("extended", "Розширена (3)", "A⁽ᵗ⁾ = f(k₁·A⁽ᵗ⁻¹⁾·W + k₂·A⁽ᵗ⁻¹⁾)", ("k1", "k2")),
    Rule("selfloop", "Із самовпливом (5)", "A⁽ᵗ⁾ = f(A⁽ᵗ⁻¹⁾·W + wᵢᵢ·A⁽ᵗ⁻¹⁾)", ("wii",)),
    Rule("quasi", "Квазінелінійна (6)", "A⁽ᵗ⁾ = φ·f(A⁽ᵗ⁻¹⁾·W) + (1 − φ)·A⁽⁰⁾", ("phi",)),
]}

DEFAULT_PARAMS = {"v": 0.5, "lam_tanh": 1.0, "lam": 1.0, "h": 0.0, "beta": 1.0,
                  "k1": 1.0, "k2": 1.0, "wii": 0.5, "phi": 0.8}

PARAM_LABELS = {"v": "поріг v (ступінчаста)", "lam_tanh": "λ (tanh)", "lam": "λ (сигмоїда)",
                "h": "зсув h (сигмоїда)", "beta": "β (експоненційна)", "k1": "k₁ (розширена)",
                "k2": "k₂ (розширена)", "wii": "wᵢᵢ (самовплив)", "phi": "φ (квазінелінійна)"}


@dataclass(frozen=True)
class Variant:
    rule: str
    act: str

    @property
    def is_baseline(self) -> bool:
        return self.rule == "native" and self.act == "none"

    def label(self, p: dict | None = None) -> str:
        p = p or DEFAULT_PARAMS
        r, a = RULES[self.rule], ACTIVATIONS[self.act]
        rp = ", ".join(f"{PARAM_LABELS[k].split(' ')[0]}={p[k]:g}" for k in r.params)
        ap = ", ".join(f"{PARAM_LABELS[k].split(' ')[0]}={p[k]:g}" for k in a.params)
        left = r.label + (f" [{rp}]" if rp else "")
        right = a.label + (f" [{ap}]" if ap else "")
        return f"{left} · {right}"


def all_variants(rules=None, acts=None) -> list[Variant]:
    """Декартів добуток правил і функцій; базовий варіант (як у моделі) завжди перший."""
    rules = list(rules or RULES)
    acts = list(acts or ACTIVATIONS)
    out = [Variant("native", "none")]
    out += [Variant(r, a) for r in RULES if r in rules for a in ACTIVATIONS if a in acts
            if not (r == "native" and a == "none")]
    return out


def simulate(W: np.ndarray, A0: np.ndarray, v: Variant, p: dict | None = None, T: int = 50,
             clamp_idx=None, blowup: float = 1e6) -> np.ndarray:
    """Траєкторії (T+1, N, n) для кожного початкового вектора A0 (N, n).

    clamp_idx — індекси концептів, які утримуються рівними A0 на кожному кроці (входи сценарію).
    Рядок, що перевищив blowup (або став нескінченним), далі заповнюється NaN (розбіжність).
    """
    p = {**DEFAULT_PARAMS, **(p or {})}
    W = np.asarray(W, float)
    A0 = np.atleast_2d(np.asarray(A0, float))
    H = np.full((T + 1, *A0.shape), np.nan)
    H[0] = A0
    A, A_prev = A0.copy(), np.zeros_like(A0)
    alive = np.ones(len(A0), bool)
    f = lambda X: activate(v.act, X, W, p)            # noqa: E731
    clamp = list(clamp_idx) if clamp_idx is not None else []
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, T + 1):
            if v.rule == "native":
                nxt = f(A + (A - A_prev) @ W)
            elif v.rule == "kosko":
                nxt = f(A @ W)
            elif v.rule == "extended":
                nxt = f(p["k1"] * (A @ W) + p["k2"] * A)
            elif v.rule == "selfloop":
                nxt = f(A @ W + p["wii"] * A)
            elif v.rule == "quasi":
                nxt = p["phi"] * f(A @ W) + (1.0 - p["phi"]) * A0
            else:
                raise KeyError(v.rule)
            if clamp:
                nxt[:, clamp] = A0[:, clamp]
            bad = ~np.all(np.isfinite(nxt), axis=1) | (np.max(np.abs(np.nan_to_num(nxt, nan=np.inf)), axis=1) > blowup)
            alive &= ~bad
            nxt[~alive] = np.nan
            A_prev, A = A, nxt
            H[t] = nxt
    return H


# ---------------------------------------------------------------- класифікація динаміки

FIXED, CYCLE, NONCONV, DIVERGED = "fixed", "cycle", "nonconv", "diverged"
STATUS_LABELS = {FIXED: "нерухома точка", CYCLE: "цикл", NONCONV: "не збіглась за T", DIVERGED: "розбіжність"}


@dataclass
class Dynamics:
    status: np.ndarray                # (N,) FIXED / CYCLE / NONCONV / DIVERGED
    steps: np.ndarray                 # (N,) крок, з якого |ΔA| < ε (для FIXED), інакше −1
    period: np.ndarray                # (N,) період циклу, інакше 0
    final: np.ndarray                 # (N, n) стан на кроці T (NaN для розбіжних)
    unique_fixed_point: bool          # усі вектори збіглися в одну точку (огляд, розд. 3)
    spread: float                     # max відстань між кінцевими станами різних векторів

    def count(self, s: str) -> int:
        return int(np.sum(self.status == s))


def classify(H: np.ndarray, eps: float = 1e-4, max_period: int = 12, unique_tol: float = 1e-4) -> Dynamics:
    """Стан кожної траєкторії за останніми кроками (огляд, розд. 3; критерій єдиності — ||Aᵢ − Aⱼ||₂ < 1e-4)."""
    T = H.shape[0] - 1
    N = H.shape[1]
    final = H[-1]
    diverged = ~np.all(np.isfinite(final), axis=1)
    d = np.max(np.abs(np.diff(H, axis=0)), axis=2)               # (T, N): max|A^t − A^{t−1}|
    status = np.full(N, NONCONV, dtype=object)
    steps = np.full(N, -1)
    period = np.zeros(N, int)
    for r in range(N):
        if diverged[r]:
            status[r] = DIVERGED
            continue
        if d[-1, r] < eps:
            status[r] = FIXED
            big = np.nonzero(d[:, r] >= eps)[0]
            steps[r] = int(big[-1]) + 2 if len(big) else 1
            continue
        for P in range(2, min(max_period, T // 2) + 1):
            if (np.max(np.abs(H[T, r] - H[T - P, r])) < eps
                    and np.max(np.abs(H[T - 1, r] - H[T - 1 - P, r])) < eps):
                status[r], period[r] = CYCLE, P
                break
    ok = final[~diverged]
    spread = 0.0
    if len(ok) > 1:
        spread = float(np.max(np.linalg.norm(ok[:, None, :] - ok[None, :, :], axis=2)))
    unique = bool(N > 1 and np.all(status == FIXED) and spread < unique_tol)
    return Dynamics(status, steps, period, final, unique, spread)


# ---------------------------------------------------------------- теоретичні умови

@dataclass
class Condition:
    name: str
    value: str
    holds: bool | None
    meaning: str


def spectral_radius(W) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(W, float))))) if np.size(W) else 0.0


def conditions(W: np.ndarray, v: Variant, p: dict | None = None, clamped: bool = False) -> list[Condition]:
    """Умови збіжності для варіанта.

    • базова формула без f: різниці D^{k+1} = D^k·W, тож процес збігається (для типових A^0) ⇔ ρ(W) < 1;
    • правила (1), (3), (5), (6) з ліпшицевою f (стала L): достатня умова стиску q = L·||M||₂ < 1 → єдина
      глобально збіжна нерухома точка (теорема Банаха; теорема 3 огляду — її частковий випадок
      для сигмоїди: ||W||_F < 4/λ, бо ||W||₂ ≤ ||W||_F);
    • сигмоїда в класичному правилі: теорема 3 (||W||_F < 4/λ) і теорема 6 огляду (λ < 4/N);
    • квазінелінійне правило з φ < 1: теорема 10 — різні початкові вектори не дають однієї точки.
    """
    p = {**DEFAULT_PARAMS, **(p or {})}
    W = np.asarray(W, float)
    n = W.shape[0]
    out: list[Condition] = []
    rho = spectral_radius(W)
    if v.rule == "native":
        if v.act == "none":
            out.append(Condition("ρ(W) < 1", f"ρ = {rho:.3f}", rho < 1,
                                 "збігається до скінченного стану" if rho < 1 else
                                 "приріст не згасає: значення можуть необмежено зростати або коливатися"))
        else:
            out.append(Condition("ρ(W)", f"ρ = {rho:.3f}", None,
                                 "для формули з приростом і функцією f загальної умови немає — лише експеримент"))
        return out
    L = lipschitz(v.act, p)
    if v.rule == "kosko":
        M, scale_ = W, 1.0
    elif v.rule == "extended":
        M, scale_ = p["k1"] * W + p["k2"] * np.eye(n), 1.0
    elif v.rule == "selfloop":
        M, scale_ = W + p["wii"] * np.eye(n), 1.0
    else:  # quasi
        M, scale_ = W, p["phi"]
    if L is not None:
        q = scale_ * L * float(np.linalg.norm(M, 2))
        if q < 1:
            if v.rule == "quasi" and p["phi"] < 1:
                meaning = "єдина нерухома точка для кожного A⁽⁰⁾; різні входи → різні точки (теорема 10)"
            elif clamped:
                meaning = "стиск: збіжність гарантована; входи утримуються, тож результат залежить від них"
            else:
                meaning = ("стиск: ОДНА точка для будь-якого A⁽⁰⁾ — модель не розрізнятиме випадки "
                           "(огляд, розд. 3)")
        else:
            meaning = "достатня умова не виконана — збіжність можлива, але не гарантована"
        out.append(Condition("стиск q = L·‖M‖₂ < 1", f"q = {q:.3f}", q < 1, meaning))
    else:
        out.append(Condition("стиск", "—", None, "функція розривна/векторна — лише експеримент (можливі цикли)"))
    if v.rule == "kosko" and v.act == "sigmoid":
        fro, lam = float(np.linalg.norm(W)), p["lam"]
        out.append(Condition("теорема 3: ‖W‖_F < 4/λ", f"{fro:.3f} < {4 / lam:.3f}", fro < 4 / lam,
                             "єдина нерухома точка" if fro < 4 / lam else "умова не виконана"))
        out.append(Condition("теорема 6: λ < 4/N", f"{lam:g} < {4 / n:.3f}", lam < 4 / n,
                             "єдина глобально стійка точка" if lam < 4 / n else "умова не виконана"))
    if v.rule == "quasi":
        out.append(Condition("теорема 10: φ < 1", f"φ = {p['phi']:g}", p["phi"] < 1,
                             "різні початкові вектори → різні точки (ін’єктивність)" if p["phi"] < 1
                             else "φ = 1: правило збігається з класичним (1)"))
    return out


# ---------------------------------------------------------------- порівняння варіантів

@dataclass
class VariantReport:
    variant: Variant
    label: str
    dyn: Dynamics
    conds: list[Condition]
    quality: dict = field(default_factory=dict)     # метрики якості з/без нормування


Scorer = Callable[[Variant, Dynamics, Callable[[np.ndarray], np.ndarray]], dict]


def compare(W, A0, variants: list[Variant], p: dict | None = None, T: int = 50, eps: float = 1e-4,
            clamp_idx=None, scorer: Scorer | None = None) -> list[VariantReport]:
    """Прогін усіх варіантів на одних і тих самих початкових векторах A0 (N, n)."""
    p = {**DEFAULT_PARAMS, **(p or {})}
    out = []
    for v in variants:
        H = simulate(W, A0, v, p, T, clamp_idx)
        dyn = classify(H, eps)
        sim = lambda X, v=v: simulate(W, X, v, p, T, clamp_idx)[-1]    # noqa: E731
        q = scorer(v, dyn, sim) if scorer else {}
        out.append(VariantReport(v, v.label(p), dyn, conditions(W, v, p, clamp_idx is not None and len(clamp_idx) > 0), q))
    return out


def spearman(a, b) -> float:
    """Ранговий коефіцієнт кореляції Спірмена (не залежить від порогів і монотонного масштабу)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = _rank(a[ok])
    rb = _rank(b[ok])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    # середні ранги для однакових значень
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def empirical_scorer(W, real_output, real_class=None, thresholds=None, out_idx: int = -1) -> Scorer:
    """Якість для моделі за даними (стаття 3): вихід — концепт out_idx на кроці T.

    «з нормуванням»: формула (11) — межі за знаками прямих дуг до виходу (як fcm.output_bounds_by_sign),
    обчислені тим самим варіантом, з обрізанням до [0, 1]; «без нормування»: сире значення виходу.
    Для класів — збіги за порогами моделі; для числа — MAD і R². Spearman ρ — для обох.
    """
    real_output = np.asarray(real_output, float)
    numeric = thresholds is None or len(thresholds) == 0

    def metrics_of(pred):
        res = {}
        ok = np.isfinite(pred)
        if not ok.all():       # хоч один випадок розбігся / межі (11) невизначені — якість не оцінюється
            return {"rs": float("nan"), "MAD": float("nan"), **({"R²": float("nan")} if numeric else {"збіги": None})}
        res["MAD"] = float(np.mean(np.abs(pred[ok] - real_output[ok])))
        res["rs"] = spearman(pred, real_class if (real_class is not None and not numeric) else real_output)
        if numeric:
            ss = float(np.sum((real_output[ok] - real_output[ok].mean()) ** 2))
            res["R²"] = 1 - float(np.sum((pred[ok] - real_output[ok]) ** 2)) / ss if ss > 0 else float("nan")
        else:
            from .metrics import classify as cls_of
            c = cls_of(np.where(ok, pred, -np.inf), thresholds)
            res["збіги"] = int(np.sum((c == real_class) & ok))
        return res

    def score(v: Variant, dyn: Dynamics, sim) -> dict:
        n = dyn.final.shape[1]
        out = out_idx % n
        raw = dyn.final[:, out]
        norm = _normalize11(raw, sim, n, out, W)
        return {"raw": raw, "norm": norm, "без": metrics_of(raw), "з": metrics_of(norm)}

    return score


def _normalize11(raw, sim, n, out, W):
    """Формула (11): (A − A_min)/(A_max − A_min); межі — входи за знаками прямих дуг до виходу."""
    W = np.asarray(W, float)
    a_min, a_max = np.zeros(n), np.zeros(n)
    for i in range(n):
        if i == out:
            continue
        if W[i, out] > 0:
            a_max[i] = 1.0
        elif W[i, out] < 0:
            a_min[i] = 1.0
    lo, hi = sim(np.vstack([a_min, a_max]))[:, out]
    if not (np.isfinite(lo) and np.isfinite(hi)) or abs(hi - lo) < 1e-6:
        return np.full_like(raw, np.nan)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def gds_scorer(Y=None, effect_idx=()) -> Scorer:
    """Якість для УДС (статті 1, 2, 4): наслідки на кроці T проти цільових Y (сценарії навчання).

    «з нормуванням»: на max|x| по всіх концептах (як fcm.normalize_maxabs, legacy); «без» — сирі значення.
    """
    effect_idx = list(effect_idx)

    def metrics_of(P):
        if Y is None:
            return {}
        ok = np.all(np.isfinite(P), axis=1)
        if not ok.all():       # хоч один сценарій розбігся — якість не оцінюється
            return {"MSE": float("nan"), "MAD": float("nan")}
        return {"MSE": float(np.mean((P[ok] - Y[ok]) ** 2)), "MAD": float(np.mean(np.abs(P[ok] - Y[ok])))}

    def score(v: Variant, dyn: Dynamics, sim) -> dict:
        F = dyn.final
        m = np.max(np.abs(F), axis=1, keepdims=True)
        Fn = np.divide(F, m, out=F.copy(), where=m > 0)
        return {"raw": F[:, effect_idx], "norm": Fn[:, effect_idx],
                "без": metrics_of(F[:, effect_idx]), "з": metrics_of(Fn[:, effect_idx])}

    return score
