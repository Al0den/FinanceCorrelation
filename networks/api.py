"""FastAPI app exposing correlation graphs as JSON for the Stock Galaxy UI."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from networks.cache import PriceCache
from networks.config import tickers as default_tickers


app = FastAPI(title="Stock Galaxy", version="0.1.0")
cache = PriceCache()


def _compute_correlation_graph(tickers: Iterable[str], window_days: int) -> Tuple[nx.Graph, pd.DataFrame]:
    today = date.today()
    start = today - timedelta(days=window_days + 5)
    prices = cache.get_prices(tickers, start.isoformat(), today.isoformat())
    if prices.empty:
        raise HTTPException(status_code=400, detail="No price data available for requested window.")
    prices = prices.sort_index().ffill()
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        raise HTTPException(status_code=400, detail="Not enough return data to compute correlations.")
    corr = returns.corr()
    graph = nx.Graph()
    graph.add_nodes_from(corr.columns)
    for i, ticker_i in enumerate(corr.columns):
        for ticker_j in corr.columns[i + 1 :]:
            weight = float(corr.loc[ticker_i, ticker_j])
            graph.add_edge(ticker_i, ticker_j, weight=weight, corr=weight, abs_corr=abs(weight))
    return graph, returns


def _communities(graph: nx.Graph) -> Dict[str, int]:
    if graph.number_of_edges() == 0:
        return {n: 0 for n in graph.nodes()}
    comms = nx_comm.greedy_modularity_communities(graph, weight="abs_corr")
    mapping: Dict[str, int] = {}
    for idx, community in enumerate(comms):
        for node in community:
            mapping[node] = idx
    return mapping or {n: 0 for n in graph.nodes()}


def _centrality_metrics(graph: nx.Graph) -> Dict[str, Dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    betweenness = nx.betweenness_centrality(graph, weight="distance") if graph.number_of_edges() else {}
    eigenvector = nx.eigenvector_centrality(graph, weight="abs_corr") if graph.number_of_edges() else {}
    metrics: Dict[str, Dict[str, float]] = {}
    for node in graph.nodes():
        metrics[node] = {
            "betweenness": float(betweenness.get(node, 0.0)),
            "eigenvector": float(eigenvector.get(node, 0.0)),
            "degree": float(graph.degree(node)),
        }
    return metrics


def _sparkline(returns: pd.DataFrame, ticker: str, points: int = 30) -> List[float]:
    if ticker not in returns.columns:
        return []
    series = returns[ticker].tail(points)
    return [float(v) for v in series.fillna(0.0)]


def _prepare_payload(graph: nx.Graph, returns: pd.DataFrame, window_days: int) -> Dict:
    metadata = cache.get_company_metadata(graph.nodes())
    for _, _, data in graph.edges(data=True):
        weight = float(data.get("corr", data.get("weight", 0.0)))
        clipped = max(-1.0, min(1.0, weight))
        data["distance"] = (2 * (1 - clipped)) ** 0.5
    centrality = _centrality_metrics(graph)
    communities = _communities(graph)
    avg_corr = returns.corr().mean()

    nodes = []
    for node in graph.nodes():
        node_meta = metadata.get(node, {"name": node, "sector": "Unknown"})
        metrics = centrality.get(node, {})
        nodes.append(
            {
                "id": node,
                "ticker": node,
                "name": node_meta.get("name", node),
                "sector": node_meta.get("sector", "Unknown"),
                "degree": metrics.get("degree", 0.0),
                "betweenness": metrics.get("betweenness", 0.0),
                "eigenvector": metrics.get("eigenvector", 0.0),
                "community": communities.get(node, 0),
                "avg_corr": float(avg_corr.get(node, 0.0)),
                "sparkline": _sparkline(returns, node),
            }
        )

    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append(
            {
                "source": u,
                "target": v,
                "corr": float(data.get("corr", 0.0)),
                "abs_corr": float(data.get("abs_corr", 0.0)),
            }
        )

    today = date.today().isoformat()
    start = (date.today() - timedelta(days=window_days)).isoformat()

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "as_of": today,
            "start": start,
            "end": today,
            "window_days": window_days,
            "tickers": list(graph.nodes()),
        },
    }


@app.get("/api/galaxy")
def galaxy(
    window_days: int = Query(90, ge=10, le=730, description="Rolling window in days for correlations"),
    tickers: List[str] | None = Query(None, description="Override default tickers"),
):
    tickers_use = tickers or default_tickers
    graph, returns = _compute_correlation_graph(tickers_use, window_days)
    payload = _prepare_payload(graph, returns, window_days)
    return payload


@app.get("/", response_class=HTMLResponse)
def homepage() -> HTMLResponse:
    index_path = StaticFiles(directory="networks/static").lookup_path("index.html")[0]
    if not index_path:
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(Path(index_path).read_text())


app.mount("/static", StaticFiles(directory="networks/static"), name="static")

