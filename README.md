# Customer Segmentation (Online Retail)

Clusters 4,338 online-retail customers by purchase behaviour (RFM plus basket features) using the UCI Online Retail dataset (541,909 transactions, 2010-2011).

## Pipeline
1. Clean: remove duplicates, missing customers, cancellations and invalid quantity/price (541,909 → 392,692 rows).
2. Features per customer: Recency, Frequency, Monetary, total quantity, unique products, tenure, average order value.
3. Log-transform + standardise, then choose k with elbow, silhouette and Davies-Bouldin.
4. Compare K-Means, Agglomerative and Gaussian Mixture.

## Results
k=2 gave the best silhouette (0.3547) and Davies-Bouldin (1.0444) scores.

| Model | Silhouette | Davies-Bouldin |
|---|---|---|
| **K-Means** | **0.3547** | **1.0444** |
| Agglomerative | 0.3193 | 1.0958 |
| Gaussian Mixture | 0.3014 | 1.1768 |

| Cluster | Customers | Recency (days) | Frequency | Monetary (£) | Avg order (£) | Unique products |
|---|---|---|---|---|---|---|
| 0: Active, high-value | 2,177 | 41 | 7.07 | 3,725 | 553 | 101 |
| 1: Lapsing, low-value | 2,161 | 144 | 1.46 | 360 | 281 | 22 |

Cluster names are my interpretation of the profiles. The model only outputs 0 or 1.

## Structure
```
notebooks/   Customer_Segmentation.ipynb
models/      customer_segmentation_model.joblib (saved by the notebook)
results/     model comparison, cluster profiles, customer segments, plots
requirements.txt
```

## Run it
1. Download the Online Retail dataset from https://archive.ics.uci.edu/dataset/352/online+retail and put `online+retail.zip` in `notebooks/`.
2. `pip install -r requirements.txt`
3. Run the notebook top to bottom.

## Use the saved model
```python
import joblib, numpy as np, pandas as pd
a = joblib.load("models/customer_segmentation_model.joblib")
cols = a["feature_columns"]
x = pd.DataFrame([[20, 8, 1500, 400, 30, 180, 187.5]], columns=cols)
print(a["model"].predict(a["scaler"].transform(np.log1p(x.clip(lower=0)))))
```

## Notes
- Only 2 clusters: the data separates into active vs lapsing customers more clearly than into finer groups.
- Silhouette is computed on a 2,000-customer sample.
- Customers without a `CustomerID` (~25% of rows) are excluded.
