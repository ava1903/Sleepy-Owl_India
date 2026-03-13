# ☕ Sleepy Owl Coffee — Cross-Sell Intelligence Dashboard

A Streamlit data analytics dashboard exploring customer cross-sell behaviour for Sleepy Owl Coffee, an Indian D2C specialty coffee brand.

## 📊 Modules

| Module | Description |
|---|---|
| 🏠 Executive Overview | Sunburst, Sankey, Waterfall — full funnel view |
| 🎯 Classification | Random Forest predicting cross-sell intent (Yes/Maybe/No) |
| 🔵 Clustering | K-Means customer personas with PCA visualisation |
| 📈 Regression | Predicting monthly spend; waterfall feature contributions |
| 🔗 Association Rules | Apriori product co-purchase rules; co-occurrence heatmap |

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set **Main file path** to `app.py`
4. Deploy

## 📁 File Structure

```
├── app.py                          # Full dashboard (single file)
├── sleepy_owl_survey_data_clean.csv # Cleaned dataset (2,000 rows)
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # Dark theme config
└── README.md
```

## 🎨 Palette

| Role | Hex |
|---|---|
| Primary background | `#0D2137` |
| Accent blue | `#1B6CA8` |
| Highlight green | `#2ECC71` |
| Secondary teal | `#17A589` |
| Muted teal | `#148F77` |

## 📦 Key Libraries

- `streamlit` — dashboard framework
- `plotly` — interactive charts (dark theme)
- `scikit-learn` — ML models
- `pandas` / `numpy` — data processing
