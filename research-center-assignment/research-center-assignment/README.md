# Research Center Quality Classification

A machine learning pipeline that classifies UK research centers into **Premium**, **Standard**, or **Basic** quality tiers using K-Means clustering, with a production-ready FastAPI endpoint and optional Docker deployment.

--- ML Pipeline for classifying UK research centers into quality tiers using K-Means clustering, with a FastAPI endpoint and Docker deployment

## Project Structure

```
research-center-assignment/
│
├── EDA_and_Model.ipynb          # Full EDA, feature selection, clustering & interpretation
├── app.py                       # FastAPI endpoint (POST /predict)
├── research_centers.csv         # Provided dataset (50 synthetic research centers)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── Dockerfile                   # Container image definition
├── docker-compose.yaml          # Compose config for local/server deployment
├── .env.draft                   # Environment variable template (safe to commit)
├── .gitignore
└── .dockerignore
```

> **Generated at runtime** (after running the notebook):  
> `cluster_model.pkl` — serialised model bundle (K-Means + scaler + feature list + tier mapping)  
> `research_centers_clustered.csv` — input data enriched with `cluster` and `qualityTier` columns

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/research-center-assignment.git
cd research-center-assignment

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

Run `EDA_and_Model.ipynb` top-to-bottom in Jupyter. This produces `cluster_model.pkl`.

```bash
jupyter notebook EDA_and_Model.ipynb
```

### 3. Start the API

```bash
uvicorn app:app --reload
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "internalFacilitiesCount": 9,
           "hospitals_10km": 3,
           "pharmacies_10km": 2,
           "facilityDiversity_10km": 0.82,
           "facilityDensity_10km": 0.45
         }'
```

**Response:**

```json
{
  "predictedCluster": 1,
  "predictedCategory": "Premium",
  "confidence": "hard-assignment (K-Means)"
}
```

---

## Docker Deployment (Bonus)

> Requires `cluster_model.pkl` to be present (train the notebook first).

```bash
# Build and run via Compose
docker compose up --build

# Or build and run manually
docker build -t rc-classifier .
docker run -p 8000:8000 rc-classifier
```

---

## Dataset

`research_centers.csv` — 50 synthetic UK research centers with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `researchCenterId` | str | Unique identifier |
| `researchCenterName` | str | Center name |
| `city` | str | City |
| `latitude`, `longitude` | float | Geographic coordinates |
| `internalFacilitiesCount` | int | Internal facility count (range 1–11) |
| `hospitals_10km` | int | Hospitals within 10 km (range 0–4) |
| `pharmacies_10km` | int | Pharmacies within 10 km (range 0–5) |
| `facilityDiversity_10km` | float | Diversity index 0–1 |
| `facilityDensity_10km` | float | Area-normalised facility density |

No missing values. All features are complete.

---

## Methodology

### Feature Selection

All five numeric features are retained:

- **`internalFacilitiesCount`** — the most direct proxy for operational capacity (highest variance, std ≈ 3.1).
- **`hospitals_10km`** / **`pharmacies_10km`** — reflect external healthcare infrastructure critical for human-subject research.
- **`facilityDiversity_10km`** — captures breadth of nearby services beyond raw counts (composite 0–1 index).
- **`facilityDensity_10km`** — area-normalised; distinguishes urban vs rural contexts.

Geographic coordinates are excluded — clustering on lat/lon would produce *location* groups, not *quality* groups.

All pairwise correlations exceed **r = 0.80**, confirming that the features share a coherent quality signal. Features are normalised with `StandardScaler` so integer counts and 0–1 indices contribute equally to Euclidean distance.

### Model Choice

K-Means (k=3) is chosen because:
- A business prior of three tiers (Premium / Standard / Basic) exists and is meaningful.
- The dataset is small (50 rows) and fully numeric — distance-based clustering is tractable.
- The elbow curve shows a clear inflection at k=3, and the silhouette score of **0.55** indicates reasonably compact, well-separated clusters.

### Tier Assignment

Clusters are ranked by composite quality score (sum of per-cluster feature means). The highest-scoring cluster maps to **Premium**, middle to **Standard**, lowest to **Basic**.

| Tier | Avg Internal Facilities | Avg Hospitals | Avg Pharmacies | Avg Diversity | Avg Density |
|------|------------------------|---------------|----------------|---------------|-------------|
| Premium | 9.5 | 3.5 | 4.1 | 0.85 | 0.54 |
| Standard | 4.9 | 1.5 | 2.1 | 0.56 | 0.29 |
| Basic | 2.3 | 0.5 | 0.4 | 0.28 | 0.13 |

---

## Discussion Points

**Which features had the greatest influence on clustering?**  
`internalFacilitiesCount` and `facilityDiversity_10km` have the highest pairwise correlations with all other features, making them the most central quality signals. However, because all five features are strongly correlated, no single variable dominates — the clustering benefits from the joint five-dimensional signal.

**What patterns were visible during EDA?**  
The data shows a clear positive gradient: centres with more internal facilities consistently have greater hospital and pharmacy access, higher diversity, and higher density. The distribution of `internalFacilitiesCount` is roughly uniform (1–11), while diversity and density are slightly right-skewed, indicating that highly-resourced centres are present but not the majority.

**Why is clustering a good approach for this problem?**  
There are no pre-existing quality labels — this is a discovery task. Unsupervised clustering lets the data define the tier boundaries objectively, rather than imposing arbitrary thresholds. K-Means is interpretable, fast, and produces stable results on this scale of data.

**How would you improve this model with real data?**  
- Enrich with real geospatial data (NHS facility registers, Ordnance Survey).  
- Add temporal features: year established, funding history, recent investments.  
- Explore DBSCAN (handles outliers) or hierarchical clustering (no fixed k required) if tier count is unknown.  
- Validate cluster stability with bootstrapping or cross-validation proxies.

**How can the endpoint be extended for continuous retraining?**  
- Trigger retraining via a scheduled pipeline (GitHub Actions cron, Apache Airflow, AWS EventBridge).  
- Version model artefacts with MLflow or DVC; serve via a model registry.  
- Add drift detection (e.g. Evidently AI) to flag when the input distribution shifts enough to warrant retraining.  
- Mount `cluster_model.pkl` as a Docker volume (already configured in `docker-compose.yaml`) so the container picks up new models without rebuilding the image.

**Bonus: How to commercialise and scale the solution?**  
1. **SaaS API** — expose the classifier as a metered REST API (per-call pricing). Research networks, NHS trusts, and real estate investors could query the tier of any facility they are evaluating.  
2. **Dashboard product** — wrap the model in a Streamlit or React frontend showing a map of tiered centers with filter/drill-down. Sell access as a subscription to procurement teams or funding bodies.  
3. **Data flywheel** — as customers submit new centers for classification, collect feedback labels to move toward a supervised model over time, improving accuracy and justifying a premium tier.  
4. **Scaling** — containerise (already done), deploy on Kubernetes (GKE/EKS) with horizontal pod autoscaling. Inference is CPU-only and sub-millisecond per request, so vertical scaling is not required until volume is very high. Place a CDN/API gateway (Kong, AWS API Gateway) in front for auth, rate limiting, and caching repeated identical queries.

---

## API Reference

### `POST /predict`

Classify a research center into a quality tier.

**Request body** (`application/json`):

```json
{
  "internalFacilitiesCount": 9,
  "hospitals_10km": 3,
  "pharmacies_10km": 2,
  "facilityDiversity_10km": 0.82,
  "facilityDensity_10km": 0.45
}
```

**Response**:

```json
{
  "predictedCluster": 1,
  "predictedCategory": "Premium",
  "confidence": "hard-assignment (K-Means)"
}
```

### `GET /health`

Liveness probe returning `{"status": "healthy"}`.

### `GET /docs`

Auto-generated Swagger UI (FastAPI built-in).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `scikit-learn` | K-Means, StandardScaler, silhouette score |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisations |
| `joblib` | Model serialisation |
| `pydantic` | Input validation |

---

## Author

**Srinidhi Radhakrishna**  
Machine Learning Engineer  
Sheffield, UK  
srinidhiriyengar22@gmail.com  
[linkedin.com/in/srinidhi-r-14a0b1230](https://linkedin.com/in/srinidhi-r-14a0b1230)
