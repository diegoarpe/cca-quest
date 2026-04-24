"""Entry point: train the strength model and simulate the 2026 World Cup."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import load_historical, load_2026
from src.model import train_strength_model, score_2026
from src.simulate import monte_carlo
from src.visualize import (
    plot_top_contenders,
    plot_round_probabilities,
    plot_feature_importance,
)


DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"


def main(n_simulations: int = 10_000) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    historical = load_historical(DATA_DIR)
    df_2026 = load_2026(DATA_DIR)

    trained = train_strength_model(historical)
    print(f"Cross-validated MAE (round_score 1-6 scale): {trained.cv_mae:.3f}")
    print("\nFeature importance:")
    print(trained.feature_importance.to_string())

    scored = score_2026(trained, df_2026)
    scored.to_csv(OUT_DIR / "team_strengths.csv", index=False)
    print("\nTop 15 teams by predicted strength:")
    print(scored.head(15).to_string(index=False))

    # Join strength back onto the feature frame so the simulator can use it.
    teams_for_sim = df_2026.merge(scored[["team", "strength"]], on="team")

    probs = monte_carlo(teams_for_sim, n_simulations=n_simulations)
    probs.to_csv(OUT_DIR / "tournament_probabilities.csv", index=False)

    pd.options.display.float_format = "{:.2%}".format
    print(f"\nMonte Carlo (N={n_simulations}) — top 20 contenders:")
    print(probs.head(20).to_string(index=False))

    champion = probs.iloc[0]
    print(
        f"\n=> Most likely champion: {champion['team']} "
        f"with P(winner) = {champion['P(winner)']:.1%}"
    )

    plot_top_contenders(probs, OUT_DIR / "top_contenders.png")
    plot_round_probabilities(probs, OUT_DIR / "round_probabilities.png")
    plot_feature_importance(trained.feature_importance, OUT_DIR / "feature_importance.png")
    print(f"\nCharts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
