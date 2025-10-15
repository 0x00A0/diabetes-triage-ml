## v0.2
**Changes**
- Model: Ridge(alpha=1.0) instead of LinearRegression
- Added high-risk classification flag (threshold = mean(y_train))
- Added Precision and Recall metrics

**Results**
| Metric | v0.1 | v0.2 | Δ |
|--------|------|------|---|
| RMSE |  56.9 | 54.7 | ↓ 3.9% |
| Precision | — | 0.61 | — |
| Recall | — | 0.59 | — |

**Rationale**
Ridge regularization reduces overfitting and yields slightly lower RMSE on held-out data.  
The added flag helps triage nurses identify high-risk patients more clearly.

---

## v0.1
- Baseline StandardScaler + LinearRegression
- API: /health, /predict
- CI/CD: full GH Actions pipeline, GHCR image publishing
