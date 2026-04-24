# World Cup 2026 — Predictive Model

Modelo predictivo que estima el ganador del Mundial 2026 combinando:

- **Ranking FIFA** actual de las 48 selecciones clasificadas
- **Factores socioeconómicos**: GDP per cápita, población, IDH
- **Factores geográficos**: continente, distancia a las sedes, ventaja de anfitrión
- **Datos históricos** de los últimos 10 mundiales (1986–2022): resultados por ronda de cada selección y las sedes en que se jugaron

## Enfoque

1. **Dataset histórico** (`data/historical_worldcups.csv`): 160 filas = 10 mundiales × 16 equipos (los 16 mejor posicionados por ronda alcanzada). Para cada equipo-torneo se registra ranking FIFA al iniciar, GDP per cápita, población, IDH, títulos previos, participaciones previas, continente, y la sede/continente anfitrión.
2. **Target** = "round score" ordinal (1=fase de grupos, 6=campeón).
3. **Feature engineering** (`src/features.py`): log-transformaciones, inverso del ranking FIFA, distancia Haversine a la sede, `elite_pedigree` (títulos × 2 + participaciones / 4), flags UEFA/CONMEBOL.
4. **Modelo** (`src/model.py`): ensemble de **Ridge regression** (60%) + **Gradient Boosting** (40%) para robustez con N≈160. La predicción se mezcla con un **prior basado en FIFA rank** para anclar el strength cuando el modelo extrapola.
5. **Validación**: Group K-Fold por año (leave-one-World-Cup-out), MAE ≈ 0.92 en escala 1–6.
6. **Simulación Monte Carlo** (`src/simulate.py`): se simulan 10 000 mundiales completos con el formato real de 48 equipos (12 grupos de 4 → R32 → R16 → cuartos → semis → final). Cada partido usa un modelo Poisson calibrado a partir del strength predicho.

## Resultado principal

Candidato más probable: **Argentina — P(campeón) ≈ 29.8%**

Top-10 con probabilidades de campeón (ejecución de referencia, N=10 000):

| Pos | Selección     | Confederación | FIFA | P(campeón) | P(final) | P(semi) |
|-----|---------------|---------------|------|------------|----------|---------|
| 1   | Argentina     | CONMEBOL      | 1    | 29.8%      | 41.0%    | 53.5%   |
| 2   | Spain         | UEFA          | 2    | 16.9%      | 27.7%    | 41.0%   |
| 3   | France        | UEFA          | 3    | 13.6%      | 24.0%    | 36.7%   |
| 4   | Italy         | UEFA          | 10   | 5.2%       | 11.7%    | 21.3%   |
| 5   | England       | UEFA          | 4    | 5.2%       | 11.8%    | 22.6%   |
| 6   | Brazil        | CONMEBOL      | 5    | 4.8%       | 10.3%    | 20.2%   |
| 7   | Germany       | UEFA          | 9    | 4.7%       | 10.1%    | 20.3%   |
| 8   | Portugal      | UEFA          | 6    | 2.7%       | 6.2%     | 14.1%   |
| 9   | Netherlands   | UEFA          | 7    | 2.2%       | 5.8%     | 13.1%   |
| 10  | Belgium       | UEFA          | 8    | 1.8%       | 4.8%     | 11.7%   |

> Resultados completos: `output/tournament_probabilities.csv`.

## Importancia de features (blend Ridge + GBM)

Las señales más informativas son, en orden:

1. `is_uefa` — UEFA/CONMEBOL han ganado los 22 mundiales.
2. `log_fifa_rank` — el ranking FIFA previo al torneo es fuertemente predictivo.
3. `elite_pedigree` — combinación de títulos y participaciones previas.
4. `is_host` — ventaja del anfitrión (modesta pero positiva).
5. `log_distance_to_host` — proximidad geográfica influye marginalmente.
6. Factores socioeconómicos (GDP, población, IDH) — aportan señal secundaria.

## Estructura del proyecto

```
world-cup-2026/
├── data/
│   ├── historical_worldcups.csv   # 10 mundiales, 160 filas
│   ├── qualified_2026.csv         # 48 selecciones clasificadas + features
│   └── hosts_2026.csv             # coordenadas de USA/MEX/CAN
├── src/
│   ├── features.py                # feature engineering
│   ├── model.py                   # Ridge + GBM ensemble
│   ├── simulate.py                # Monte Carlo del bracket de 48 equipos
│   └── visualize.py               # charts
├── output/
│   ├── team_strengths.csv
│   ├── tournament_probabilities.csv
│   ├── top_contenders.png
│   ├── round_probabilities.png
│   └── feature_importance.png
├── main.py
└── world_cup_2026_colab.ipynb     # notebook autocontenido para Colab
```

## Cómo correrlo

### Local

```bash
pip install scikit-learn pandas numpy matplotlib
cd world-cup-2026
python3 main.py
```

### Google Colab

Subí `world_cup_2026_colab.ipynb` a Colab y ejecutá todas las celdas. El notebook descarga los CSVs desde el repo y corre el pipeline completo en ~30 s.

## Limitaciones conocidas

- **Muestra chica**: 10 mundiales × 16 equipos no es un dataset grande. Por eso usamos Ridge + prior basado en rank.
- **Datos 2026**: el ranking FIFA y los 48 clasificados se estiman a la fecha de ejecución; reemplazar `data/qualified_2026.csv` con valores oficiales antes del sorteo cambia las probabilidades.
- **Modelo de partido**: Poisson simple con un factor de strength. No modela estilo, lesiones, forma reciente ni jugadores individuales.
- **Draw aleatorio**: los bombos se arman por ranking; el sorteo real introduce varianza adicional (grupos de la muerte).

## Ética

El modelo **no predice** el resultado real — produce distribuciones de probabilidad basadas en señales públicas. Incluso el equipo con mayor probabilidad (~30%) tiene 70% de chance de NO ganar el torneo. Tratalo como análisis, no como pronóstico firme.
