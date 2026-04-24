"""Monte Carlo simulation of the 48-team World Cup 2026 bracket.

Strengths from the strength model are converted to goal expectations with a
Poisson model, then matches are simulated. Group stage -> Round of 32 ->
Round of 16 -> Quarterfinals -> Semifinals -> Final.

Notes on the 2026 format:
- 48 teams split into 12 groups of 4.
- Top 2 per group + 8 best third-placed qualify (32 teams advance).
- From the Round of 32 onwards it is single-elimination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# How strength (model output roughly in 1..6) maps to expected goals per match.
# Calibrated so the best team averages ~2.2 goals and the weakest ~0.5.
STRENGTH_TO_XG_SLOPE = 0.45
STRENGTH_TO_XG_INTERCEPT = -0.5


def strength_to_xg(strength: float) -> float:
    return max(0.3, STRENGTH_TO_XG_SLOPE * strength + 1.0 + STRENGTH_TO_XG_INTERCEPT)


def simulate_match(
    rng: np.random.Generator,
    s_a: float,
    s_b: float,
    must_have_winner: bool,
) -> Tuple[int, int, int]:
    """Return (goals_a, goals_b, winner_idx) where winner_idx in {0,1,-1}."""
    mean_a = strength_to_xg(s_a) * (1 + 0.12 * (s_a - s_b))
    mean_b = strength_to_xg(s_b) * (1 + 0.12 * (s_b - s_a))
    mean_a = max(0.2, mean_a)
    mean_b = max(0.2, mean_b)

    goals_a = rng.poisson(mean_a)
    goals_b = rng.poisson(mean_b)

    if goals_a > goals_b:
        return goals_a, goals_b, 0
    if goals_b > goals_a:
        return goals_a, goals_b, 1

    if not must_have_winner:
        return goals_a, goals_b, -1

    # Penalty shootout: lightly favor the stronger side.
    p_a = 0.5 + 0.07 * (s_a - s_b)
    p_a = float(np.clip(p_a, 0.15, 0.85))
    winner = 0 if rng.random() < p_a else 1
    return goals_a, goals_b, winner


def assign_groups(
    rng: np.random.Generator,
    teams: pd.DataFrame,
) -> List[List[str]]:
    """Pot-based draw into 12 groups of 4.

    - Pot 1: 3 hosts + 9 highest-ranked non-hosts
    - Pots 2-4: remaining teams split by FIFA rank
    """
    teams = teams.copy().sort_values("fifa_rank").reset_index(drop=True)
    hosts = teams[teams["host"] == 1]["team"].tolist()
    non_hosts = teams[teams["host"] == 0]["team"].tolist()

    pot1 = hosts + non_hosts[: 12 - len(hosts)]
    rest = [t for t in teams["team"].tolist() if t not in pot1]
    pot2 = rest[:12]
    pot3 = rest[12:24]
    pot4 = rest[24:36]

    for pot in (pot2, pot3, pot4):
        rng.shuffle(pot)

    groups: List[List[str]] = [[pot1[i]] for i in range(12)]
    for pot in (pot2, pot3, pot4):
        for i in range(12):
            groups[i].append(pot[i])
    return groups


def simulate_group(
    rng: np.random.Generator,
    group: List[str],
    strength: Dict[str, float],
) -> List[Tuple[str, int, int, int, int]]:
    """Return standings: list of (team, points, gd, gf, pos)."""
    stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in group}
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            ga, gb, w = simulate_match(rng, strength[a], strength[b], must_have_winner=False)
            stats[a]["gf"] += ga
            stats[a]["ga"] += gb
            stats[b]["gf"] += gb
            stats[b]["ga"] += ga
            if w == 0:
                stats[a]["pts"] += 3
            elif w == 1:
                stats[b]["pts"] += 3
            else:
                stats[a]["pts"] += 1
                stats[b]["pts"] += 1

    standings = sorted(
        stats.items(),
        key=lambda kv: (kv[1]["pts"], kv[1]["gf"] - kv[1]["ga"], kv[1]["gf"]),
        reverse=True,
    )
    return [
        (team, s["pts"], s["gf"] - s["ga"], s["gf"], pos)
        for pos, (team, s) in enumerate(standings)
    ]


def knockout_round(
    rng: np.random.Generator,
    pairings: List[Tuple[str, str]],
    strength: Dict[str, float],
) -> List[str]:
    winners: List[str] = []
    for a, b in pairings:
        _, _, w = simulate_match(rng, strength[a], strength[b], must_have_winner=True)
        winners.append(a if w == 0 else b)
    return winners


@dataclass
class SimulationResult:
    winners: List[str] = field(default_factory=list)
    finalists: List[str] = field(default_factory=list)
    semifinalists: List[str] = field(default_factory=list)
    quarterfinalists: List[str] = field(default_factory=list)


def run_tournament(
    rng: np.random.Generator,
    teams: pd.DataFrame,
    result: SimulationResult,
) -> None:
    strength = dict(zip(teams["team"], teams["strength"]))
    groups = assign_groups(rng, teams)

    advanced: List[str] = []  # top-2 per group
    thirds: List[Tuple[str, int, int, int]] = []  # (team, pts, gd, gf)
    for g in groups:
        standings = simulate_group(rng, g, strength)
        advanced.append(standings[0][0])
        advanced.append(standings[1][0])
        thirds.append((standings[2][0], standings[2][1], standings[2][2], standings[2][3]))

    # Best 8 third-placed teams complete the Round of 32.
    thirds.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    advanced.extend([t[0] for t in thirds[:8]])
    rng.shuffle(advanced)

    # Round of 32 -> 16
    r32_pairings = [(advanced[i], advanced[i + 1]) for i in range(0, 32, 2)]
    r16 = knockout_round(rng, r32_pairings, strength)

    r16_pairings = [(r16[i], r16[i + 1]) for i in range(0, 16, 2)]
    qf = knockout_round(rng, r16_pairings, strength)
    result.quarterfinalists.extend(qf)

    qf_pairings = [(qf[i], qf[i + 1]) for i in range(0, 8, 2)]
    sf = knockout_round(rng, qf_pairings, strength)
    result.semifinalists.extend(sf)

    sf_pairings = [(sf[0], sf[1]), (sf[2], sf[3])]
    finalists = knockout_round(rng, sf_pairings, strength)
    result.finalists.extend(finalists)

    champion = knockout_round(rng, [(finalists[0], finalists[1])], strength)[0]
    result.winners.append(champion)


def monte_carlo(
    teams: pd.DataFrame,
    n_simulations: int = 10_000,
    seed: int = 2026,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = SimulationResult()
    for _ in range(n_simulations):
        run_tournament(rng, teams, result)

    def _count(series: List[str]) -> Dict[str, int]:
        s = pd.Series(series).value_counts()
        return s.to_dict()

    wins = _count(result.winners)
    finals = _count(result.finalists)
    semis = _count(result.semifinalists)
    quarters = _count(result.quarterfinalists)

    rows = []
    for team in teams["team"]:
        rows.append({
            "team": team,
            "P(winner)": wins.get(team, 0) / n_simulations,
            "P(final)": finals.get(team, 0) / n_simulations,
            "P(semifinal)": semis.get(team, 0) / n_simulations,
            "P(quarterfinal)": quarters.get(team, 0) / n_simulations,
        })
    return pd.DataFrame(rows).sort_values("P(winner)", ascending=False).reset_index(drop=True)
