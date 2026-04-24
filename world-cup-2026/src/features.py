"""Feature engineering for the World Cup 2026 prediction model.

Combines FIFA ranking, socioeconomic indicators (GDP per capita, population, HDI),
geographic factors (continent match with host, distance to host city, host
advantage) and historical factors (previous titles, previous participations).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_TO_SCORE = {
    "Group": 1,
    "Round16": 2,
    "Quarterfinal": 3,
    "Semifinal": 4,
    "Final": 5,
    "Winner": 6,
}

# Approximate coordinates for the capitals of past host countries.
HOST_COORDS = {
    "Mexico": (19.43, -99.13),
    "Italy": (41.90, 12.50),
    "USA": (38.90, -77.04),
    "France": (48.86, 2.35),
    "South Korea": (37.57, 126.98),
    "Germany": (52.52, 13.41),
    "South Africa": (-25.75, 28.19),
    "Brazil": (-15.78, -47.93),
    "Russia": (55.75, 37.62),
    "Qatar": (25.29, 51.53),
}

# Typical confederation -> continent bucket used for host_continent features.
CONF_TO_CONTINENT = {
    "CONMEBOL": "South America",
    "CONCACAF": "North America",
    "UEFA": "Europe",
    "CAF": "Africa",
    "AFC": "Asia",
    "OFC": "Oceania",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


TEAM_COORDS = {
    "Argentina": (-34.6, -58.4), "Brazil": (-15.8, -47.9), "Uruguay": (-34.9, -56.2),
    "Paraguay": (-25.3, -57.6), "Chile": (-33.4, -70.6), "Colombia": (4.7, -74.1),
    "Ecuador": (-0.2, -78.5), "Peru": (-12.0, -77.0), "Bolivia": (-16.5, -68.1),
    "Venezuela": (10.5, -66.9),
    "United States": (38.9, -77.0), "Mexico": (19.4, -99.1), "Canada": (45.4, -75.7),
    "Costa Rica": (9.9, -84.1), "Honduras": (14.1, -87.2), "Panama": (8.9, -79.5),
    "Jamaica": (18.1, -76.8),
    "France": (48.8, 2.3), "Germany": (52.5, 13.4), "West Germany": (50.1, 8.7),
    "Spain": (40.4, -3.7), "Italy": (41.9, 12.5), "England": (51.5, -0.1),
    "Netherlands": (52.4, 4.9), "Portugal": (38.7, -9.1), "Belgium": (50.8, 4.4),
    "Croatia": (45.8, 15.9), "Switzerland": (46.9, 7.4), "Denmark": (55.7, 12.6),
    "Sweden": (59.3, 18.1), "Norway": (59.9, 10.7), "Poland": (52.2, 21.0),
    "Austria": (48.2, 16.4), "Czechoslovakia": (50.1, 14.4), "Yugoslavia": (44.8, 20.5),
    "Serbia": (44.8, 20.5), "Bulgaria": (42.7, 23.3), "Romania": (44.4, 26.1),
    "Greece": (37.98, 23.73), "Turkey": (39.9, 32.9), "Ireland": (53.3, -6.3),
    "Republic of Ireland": (53.3, -6.3), "Ukraine": (50.5, 30.5), "Russia": (55.8, 37.6),
    "Soviet Union": (55.8, 37.6), "Slovakia": (48.1, 17.1), "Scotland": (55.9, -3.2),
    "Morocco": (34.0, -6.8), "Egypt": (30.0, 31.2), "Algeria": (36.8, 3.1),
    "Tunisia": (36.8, 10.2), "Senegal": (14.7, -17.5), "Nigeria": (9.1, 7.5),
    "Cameroon": (3.9, 11.5), "Ghana": (5.6, -0.2), "Ivory Coast": (7.5, -5.5),
    "Japan": (35.7, 139.7), "South Korea": (37.5, 127.0), "Iran": (35.7, 51.4),
    "Saudi Arabia": (24.7, 46.7), "Qatar": (25.3, 51.5), "Iraq": (33.3, 44.4),
    "Australia": (-35.3, 149.1), "Uzbekistan": (41.3, 69.2), "Jordan": (31.9, 35.9),
    "New Zealand": (-41.3, 174.8),
}


def distance_to_host_km(team: str, host: str) -> float:
    team_xy = TEAM_COORDS.get(team)
    host_xy = HOST_COORDS.get(host)
    if team_xy is None or host_xy is None:
        return 8000.0  # fallback median intercontinental distance
    return haversine_km(team_xy[0], team_xy[1], host_xy[0], host_xy[1])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a frame of team-tournament rows."""
    out = df.copy()

    # Host flag: 1 if team IS the host, else 0.
    out["is_host"] = (out["team"] == out["host"]).astype(int)

    # Continent matches host continent
    out["same_continent_host"] = (out["continent"] == out["host_continent"]).astype(int)

    # Distance to host capital.
    out["distance_to_host_km"] = out.apply(
        lambda r: distance_to_host_km(r["team"], r["host"]), axis=1,
    )

    # Log transforms for skewed numeric variables.
    out["log_gdp_per_capita"] = np.log1p(out["gdp_per_capita_usd"])
    out["log_population"] = np.log1p(out["population_millions"])
    out["log_distance_to_host"] = np.log1p(out["distance_to_host_km"])

    # FIFA rank transformations: lower rank number = stronger team.
    out["inv_fifa_rank"] = 1.0 / out["fifa_rank_at_start"].clip(lower=1)
    out["log_fifa_rank"] = np.log1p(out["fifa_rank_at_start"])

    # Elite pedigree: a simple mix of titles and repeated appearances.
    out["elite_pedigree"] = out["prior_titles"] * 2 + out["prior_participations"] / 4.0

    # UEFA / CONMEBOL flags: these two confederations have won every WC.
    out["is_uefa"] = (out["continent"] == "UEFA").astype(int)
    out["is_conmebol"] = (out["continent"] == "CONMEBOL").astype(int)

    return out


def engineer_features_2026(
    qualified: pd.DataFrame,
    hosts: pd.DataFrame,
) -> pd.DataFrame:
    """Shape the 2026 qualified teams into the same feature space."""
    out = qualified.copy()

    # Map confederation -> continent bucket compatible with the historical encoding.
    out["continent_bucket"] = out["confederation"].map(CONF_TO_CONTINENT).fillna("Europe")

    # Host continent for 2026 is North America (all three hosts are CONCACAF).
    out["host_continent"] = "CONCACAF"
    out["same_continent_host"] = (out["confederation"] == "CONCACAF").astype(int)
    out["is_host"] = out["host"].astype(int)

    # Distance to the nearest 2026 host capital.
    def nearest_host_km(row: pd.Series) -> float:
        dists = [
            haversine_km(row["latitude"], row["longitude"], h["latitude"], h["longitude"])
            for _, h in hosts.iterrows()
        ]
        return min(dists)

    out["distance_to_host_km"] = out.apply(nearest_host_km, axis=1)

    out["log_gdp_per_capita"] = np.log1p(out["gdp_per_capita_usd"])
    out["log_population"] = np.log1p(out["population_millions"])
    out["log_distance_to_host"] = np.log1p(out["distance_to_host_km"])
    out["inv_fifa_rank"] = 1.0 / out["fifa_rank"].clip(lower=1)
    out["log_fifa_rank"] = np.log1p(out["fifa_rank"])
    out["fifa_rank_at_start"] = out["fifa_rank"]
    out["elite_pedigree"] = out["prior_titles"] * 2 + out["prior_participations"] / 4.0
    out["is_uefa"] = (out["confederation"] == "UEFA").astype(int)
    out["is_conmebol"] = (out["confederation"] == "CONMEBOL").astype(int)

    return out


def load_historical(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "historical_worldcups.csv")
    df["round_score"] = df["final_round"].map(ROUND_TO_SCORE)
    return engineer_features(df)


def load_2026(data_dir: Path) -> pd.DataFrame:
    qualified = pd.read_csv(data_dir / "qualified_2026.csv")
    hosts = pd.read_csv(data_dir / "hosts_2026.csv")
    return engineer_features_2026(qualified, hosts)


FEATURE_COLUMNS = [
    "inv_fifa_rank",
    "log_fifa_rank",
    "log_gdp_per_capita",
    "log_population",
    "hdi",
    "elite_pedigree",
    "prior_titles",
    "prior_participations",
    "is_host",
    "same_continent_host",
    "log_distance_to_host",
    "is_uefa",
    "is_conmebol",
]
