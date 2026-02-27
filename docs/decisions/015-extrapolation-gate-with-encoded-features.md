# ADR-015: Extrapolation Gate with Encoded Features and KD-Tree

## Status

Accepted

## Date

2026-02-26

## Context

Surrogate models are unreliable outside the training data region. The optimizer can propose candidate points in unexplored regions of the design space where surrogate predictions are meaningless. An extrapolation gate penalizes such candidates, keeping the optimizer within the data-supported region.

The gate must handle mixed variable types: continuous, integer, categorical, and ordinal. The challenge is defining "distance" for categorical variables that have no natural metric.

### Options

**Option A — Gower distance with custom sklearn metric:** Gower distance handles mixed types natively (range-normalized Manhattan for numeric, simple matching for categorical). However, sklearn's `NearestNeighbors` with a custom callable metric falls back to brute-force (no KD-Tree or Ball-Tree). With population_size=100 and n_generations=200, that is 20,000 queries, each O(n) against the full training set. For large datasets (tens of thousands of rows, realistic at TenneT), this becomes a bottleneck.

**Option B — Encode to numeric, use KD-Tree:** Encode all variables to numeric (one-hot for categorical, positional integer for ordinal), range-normalize to [0, 1], then use sklearn's KD-Tree with Euclidean distance. KD-Tree gives O(log n) queries. The encoding is applied once to training data and once per generation to candidates.

**Option C — Separate numeric and categorical gates:** Run k-NN on numeric features only, then separately check if categorical feature combinations exist in training data. Two separate notions of "extrapolation" that are hard to combine into a single penalty.

## Decision

Option B: Encode all decision variable types to numeric, range-normalize, use KD-Tree.

Encoding rules:
- **Continuous / Integer**: pass through (already numeric)
- **Categorical**: one-hot encoding via `pd.get_dummies` (each category becomes a binary column)
- **Ordinal**: integer encoding by position in `bounds.categories` (0, 1, 2, ...). Preserves declared order.
- **Range normalization**: all features scaled to [0, 1] using training data min/max. Ensures all dimensions contribute equally to distance.

Threshold semantics:
- During init, compute k-NN distances for all training points to their own k nearest neighbors. `median_knn_distance = np.median(all_knn_distances)` — the "typical density" of the training data.
- A candidate is flagged as extrapolating when its mean k-NN distance exceeds `threshold * median_knn_distance` (default threshold=2.0: "more than 2x the typical training density").
- Flagged points receive a penalty on all objectives: `penalty = 100 * max(objective_range)` computed from training data. This scales correctly regardless of whether objectives are in cents (millions) or normalized [0,1].

## Rationale

- **Performance**: KD-Tree is O(log n) per query vs O(n) for brute-force Gower. At 20,000 queries against 10,000 training points, this is ~270x faster.
- **One-hot for categorical**: A novel categorical combination (e.g., "4-Schicht" + "Automatik" never seen together) produces a one-hot vector that is geometrically distant from all training vectors. The k-NN distance naturally captures this — no separate categorical-combination check needed.
- **Ordinal as positional integer (intentionally different from ADR-014)**: ADR-014 treats ordinal as categorical in the *search space* to avoid implying equal step sizes in mutation/crossover operators. Here in the *distance space*, positional integer encoding preserves the declared ordering, which helps find closer training neighbors (e.g., "medium" is genuinely closer to "low" than "high" is). This inconsistency is deliberate: in the search space, equal spacing causes wrong operator behavior; in the distance space, ordering information improves neighbor detection.

## Consequences

- One-hot encoding increases dimensionality. A categorical variable with 10 categories adds 10 columns. KD-Tree performance degrades above ~15 dimensions (e.g., three categorical variables with 5 categories each = 15 one-hot columns). **Automatic fallback**: if encoded dimensionality exceeds 20, the implementation switches from KD-Tree to BallTree, which handles higher dimensions better. Above ~50 dimensions, BallTree also degrades, but this requires 10+ high-cardinality categoricals, which is exotic for industrial optimization.
- The encoding must be applied identically to training data and candidates. The `ExtrapolationGate` stores the encoding parameters (min/max for normalization, category-to-column mapping) and applies them consistently.
- Unseen categorical values in candidates (not in training categories) are handled by the variable bounds — pymoo only proposes values within declared bounds.
