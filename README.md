# Stock Galaxy – Correlation Network Explorer

Interactive force-directed explorer for equity correlations. A FastAPI backend computes rolling correlations, centrality metrics, and community labels, while a lightweight D3 UI lets you filter by window length and correlation strength.

## Running the app

```bash
pip install -r requirements.txt
uvicorn networks.api:app --reload
```

Then open http://127.0.0.1:8000/ to explore the galaxy. The D3 view colors nodes by sector, sizes them by degree, and adds halos for high betweenness centrality. Use the sliders to adjust lookback window and minimum correlation; the graph updates live without a page refresh.

## API

- `GET /api/galaxy?window_days=90&tickers=AAPL&tickers=MSFT` returns nodes, edges, and metadata for the requested window. Results include sector labels, average correlations, sparkline-ready returns, Louvain-style communities, and betweenness/eigenvector scores.
