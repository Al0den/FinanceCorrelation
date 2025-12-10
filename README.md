# Stock Galaxy – Correlation Network Explorer

Interactive force-directed explorer for equity correlations. A FastAPI backend computes rolling correlations, centrality metrics, and community labels, while a lightweight D3 UI lets you filter by window length and correlation strength.

## Running the app

```bash
pip install -r requirements.txt
uvicorn networks.api:app --reload
```

Then open http://127.0.0.1:8000/ to explore the galaxy. The D3 view colors nodes by sector, sizes them by degree, and adds halos for high betweenness centrality. Use the sliders to adjust lookback window and minimum correlation; the graph updates live without a page refresh. You can also curate the ticker universe inline—paste comma-separated symbols, remove chips, and reload on the fly to compare bespoke baskets.

## API

- `GET /api/galaxy?window_days=90&tickers=AAPL&tickers=MSFT` returns nodes, edges, and metadata for the requested window. Results include sector labels, average correlations, sparkline-ready returns, Louvain-style communities, and betweenness/eigenvector scores.
- `GET /api/tickers` surfaces the curated default ticker set along with cached company metadata for use in client pickers.
