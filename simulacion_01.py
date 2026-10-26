import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Parámetros Globales
# ===============================
SIMS_PER_MASK = 300
SEED = None

if SEED is not None:
    random.seed(SEED)

CATEGORIES_ORDER = [
    "ones","twos","threes","fours","fives","sixes",
    "threeOfKind","fourOfKind","fullHouse","smallStraight",
    "largeStraight","yahtzee","chance"
]

# ===============================
# Motor Físico (Dados y Puntuación)
# ===============================
def roll_single_die() -> int:
    return random.randint(1, 6)

def roll_dice(values: List[int], locked: List[bool]) -> None:
    for i, is_locked in enumerate(locked):
        if not is_locked:
            values[i] = roll_single_die()

def has_sequence(values: List[int], needed_len: int) -> bool:
    uniq = sorted(set(values))
    if not uniq: return False
    longest = curr = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i-1] + 1:
            curr += 1
            longest = max(longest, curr)
        else: curr = 1
    return longest >= needed_len

def score_hand(category: str, dice: List[int]) -> int:
    counts = Counter(dice)
    freqs = list(counts.values())
    total = sum(dice)
    if category in ["ones","twos","threes","fours","fives","sixes"]:
        num = {"ones":1, "twos":2, "threes":3, "fours":4, "fives":5, "sixes":6}[category]
        return counts.get(num, 0) * num
    if category == "threeOfKind": return total if any(f >= 3 for f in freqs) else 0
    if category == "fourOfKind": return total if any(f >= 4 for f in freqs) else 0
    if category == "fullHouse": return 25 if (2 in freqs and 3 in freqs) else 0
    if category == "smallStraight": return 30 if has_sequence(dice, 4) else 0
    if category == "largeStraight": return 40 if has_sequence(dice, 5) else 0
    if category == "yahtzee": return 50 if 5 in freqs else 0
    if category == "chance": return total
    return 0

def best_category_score(dice: List[int], available: List[str]) -> Tuple[str, int]:
    best_cat, best = None, -1
    for cat in available:
        sc = score_hand(cat, dice)
        if sc > best:
            best = sc
            best_cat = cat
    return best_cat, best

# ===============================
# Estructuras de Estado
# ===============================
@dataclass
class PlayerState:
    name: str
    scores: Dict[str, Optional[int]] = field(default_factory=lambda: {c: None for c in CATEGORIES_ORDER})
    total: int = 0
    def available_categories(self) -> List[str]:
        return [c for c,v in self.scores.items() if v is None]
    def set_score(self, category: str, points: int) -> None:
        self.scores[category] = points
        self.total = sum(v for v in self.scores.values() if v is not None)

@dataclass
class GameStats:
    total_rolls: int = 0
    face_hist: Counter = field(default_factory=Counter)
    def update_from_roll(self, values: List[int]) -> None:
        self.total_rolls += len(values)
        self.face_hist.update(values)

@dataclass
class GameState:
    players: List[PlayerState]
    current_player_idx: int = 0
    dice_values: List[int] = field(default_factory=lambda: [0]*5)
    locked: List[bool] = field(default_factory=lambda: [False]*5)
    rolls_left: int = 3
    stats: GameStats = field(default_factory=GameStats)

    def reset_turn(self):
        self.dice_values = [0]*5
        self.locked = [False]*5
        self.rolls_left = 3
    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]
    def advance_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

# ===============================
# El Cerebro: Montecarlo
# ===============================
def all_keep_masks() -> List[List[bool]]:
    return [[(m >> i) & 1 == 1 for i in range(5)] for m in range(32)]

def simulate_expected_value_after_keeps(current_values: List[int], keep_mask: List[bool], rolls_remaining: int, available_categories: List[str], sims: int) -> float:
    if rolls_remaining <= 0:
        _, sc = best_category_score(current_values, available_categories)
        return float(sc)

    kept = [v if keep_mask[i] else None for i, v in enumerate(current_values)]
    unlocked_idx = [i for i,b in enumerate(keep_mask) if not b]
    total_score = 0.0

    for _ in range(sims):
        vals = kept[:]
        temp = current_values[:]
        for _r in range(rolls_remaining):
            for i in unlocked_idx:
                temp[i] = roll_single_die()
        for i in range(5):
            vals[i] = vals[i] if vals[i] is not None else temp[i]
        _, sc = best_category_score(vals, available_categories)
        total_score += sc

    return total_score / sims if sims > 0 else 0.0

def choose_keep_mask_montecarlo(current_values: List[int], rolls_remaining: int, available_categories: List[str], sims_per_mask: int = SIMS_PER_MASK, verbose: bool = False) -> List[bool]:
    resultados = []
    for mask in all_keep_masks():
        ev = simulate_expected_value_after_keeps(current_values, mask, rolls_remaining - 1, available_categories, sims_per_mask)
        resultados.append((mask, ev))

    resultados.sort(key=lambda x: x[1], reverse=True)
    best_mask, best_ev = resultados[0]

    if verbose:
        print(f"   [MONTECARLO] Evaluando 32 escenarios posibles ({sims_per_mask} sims/escenario)...")
        print(f"   🌟 Mejor Opción -> Máscara: {best_mask} | EV: {best_ev:.2f} pts")
        print(f"   🔸 Opción 2     -> Máscara: {resultados[1][0]} | EV: {resultados[1][1]:.2f} pts")

    return best_mask

# ===============================
# Flujo del Juego
# ===============================
def play_one_turn(state: GameState, verbose: bool = False):
    player = state.current_player
    state.reset_turn()
    roll_dice(state.dice_values, state.locked)
    state.stats.update_from_roll(state.dice_values)
    state.rolls_left -= 1

    if verbose: print(f"\n{'-'*40}\n🎲 Turno de {player.name} | Tirada 1: {state.dice_values}")

    while state.rolls_left > 0:
        mostrar_mente_ia = verbose and state.rolls_left == 2
        keep_mask = choose_keep_mask_montecarlo(state.dice_values, state.rolls_left, player.available_categories(), SIMS_PER_MASK, verbose=mostrar_mente_ia)

        if all(keep_mask):
            if verbose: print(f"   => {player.name} se PLANTA.")
            break

        state.locked = keep_mask[:]
        roll_dice(state.dice_values, state.locked)
        state.stats.update_from_roll([v for i, v in enumerate(state.dice_values) if not keep_mask[i]])
        state.rolls_left -= 1

        if verbose: print(f"🎲 Tirada {3 - state.rolls_left}: {state.dice_values}")

    cat, sc = best_category_score(state.dice_values, player.available_categories())
    player.set_score(cat, sc)
    if verbose: print(f"📝 {player.name} anota {sc} pts en la categoría '{cat}'. (Total: {player.total})")

def play_game():
    state = GameState(players=[PlayerState("IA Montecarlo 1"), PlayerState("IA Montecarlo 2")])
    print("\n🚀 INICIANDO PARTIDA YAHTZEE...")
    for ronda in range(1, 14):
        print(f"\n{'='*15} RONDA {ronda} {'='*15}")
        for _ in range(2):
            play_one_turn(state, verbose=True)
            state.advance_player()

    print("\n🏆 RESULTADOS FINALES 🏆")
    p1, p2 = state.players
    print(f"{p1.name}: {p1.total} pts | {p2.name}: {p2.total} pts")
    return state

# ===============================
# Módulo de Visualización (NUEVO)
# ===============================
def mostrar_analisis_grafico(estado: GameState):
    sns.set_theme(style="whitegrid")

    # 1. Gráfica de Distribución de Caras de los Dados
    total_faces = sum(estado.stats.face_hist.values())
    caras = list(range(1, 7))
    probs = [(estado.stats.face_hist.get(f, 0) / total_faces) * 100 for f in caras]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=caras, y=probs, color="skyblue")
    plt.axhline(y=16.67, color='red', linestyle='--', linewidth=2, label='Teórica (16.67%)')
    plt.title(f"Distribución de Caras tras {total_faces} dados lanzados", fontweight="bold", fontsize=12)
    plt.xlabel("Cara del Dado")
    plt.ylabel("Frecuencia Relativa (%)")

    # Añadir porcentajes sobre las barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Experimento Analítico: EV vs Costo Computacional
    print("\n" + "="*50)
    print("🔬 ESTUDIO: CONVERGENCIA (LLN) vs COSTO COMPUTACIONAL")
    print("="*50)

    dados_prueba = [3, 3, 5, 2, 6]
    mascara_prueba = [True, True, False, False, False]
    categorias = CATEGORIES_ORDER[:]

    # Rango de simulaciones a evaluar
    muestras_n = [10, 50, 100, 300, 500, 1000, 3000, 5000]
    ev_calculados = []
    tiempos = []

    for n in muestras_n:
        t_inicio = time.time()
        ev = simulate_expected_value_after_keeps(dados_prueba, mascara_prueba, 2, categorias, sims=n)
        t_fin = time.time() - t_inicio
        ev_calculados.append(ev)
        tiempos.append(t_fin)
        print(f"   Iteraciones (N): {n:<5} | EV: {ev:.2f} pts | Tiempo: {t_fin:.4f} seg")

    # Panel Doble (Subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfica A: Convergencia (LLN)
    ax1.plot(muestras_n, ev_calculados, marker='o', color='purple', linewidth=2)
    ax1.axhline(y=ev_calculados[-1], color='red', linestyle='--', label=f'EV Convergente (~{ev_calculados[-1]:.2f})')
    ax1.set_title("Teoría de los Grandes Números (Precisión)", fontweight="bold")
    ax1.set_xlabel("Iteraciones (N) - Escala Logarítmica")
    ax1.set_ylabel("Valor Esperado Estimado (EV)")
    ax1.set_xscale('log')
    ax1.legend()

    # Gráfica B: Costo Computacional
    ax2.plot(muestras_n, tiempos, marker='s', color='orange', linewidth=2)
    ax2.set_title("Costo Computacional vs Iteraciones", fontweight="bold")
    ax2.set_xlabel("Iteraciones (N) - Escala Logarítmica")
    ax2.set_ylabel("Tiempo de Ejecución (Segundos)")
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.show()

# ===============================
# EJECUCIÓN PRINCIPAL
# ===============================
if __name__ == "__main__":
    estado_final = play_game()
    mostrar_analisis_grafico(estado_final)