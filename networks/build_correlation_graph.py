"""
Build a correlation graph for the tickers defined in `networks/config.py`.

The script downloads price data for the configured date range, computes daily
returns, constructs an undirected correlation graph with correlation
coefficients as edge weights, and saves it to the `data/` directory. The output
filename follows the pattern:

    <start_date>-<end_date>-<tickers_hash>.graphml

where `tickers_hash` is an 8-digit hash of the sorted tickers, so runs with the
same inputs are deterministic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

from networks import config
from networks.cache import PriceCache


def _hash_tickers(tickers: Iterable[str]) -> str:
    """Return an 8-digit numeric hash for the given tickers."""
    joined = ",".join(sorted(t.strip().upper() for t in tickers if t.strip()))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    numeric = int(digest, 16) % 10**8
    return f"{numeric:08d}"


def _build_correlation_graph(prices: pd.DataFrame) -> nx.Graph:
    """Create an undirected graph with correlation coefficients as edge weights."""
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    corr = returns.corr()
    graph = nx.Graph()
    graph.add_nodes_from(corr.columns)
    for i, ticker_i in enumerate(corr.columns):
        for ticker_j in corr.columns[i + 1 :]:
            weight = corr.loc[ticker_i, ticker_j]
            graph.add_edge(ticker_i, ticker_j, weight=weight)
    return graph


def _sanitize_for_graphml(graph: nx.Graph) -> None:
    """Ensure all attributes are GraphML-safe (strings/numbers/bools)."""
    def _coerce(value):
        if isinstance(value, (str, int, float, bool)):
            return value
        try:
            import pandas as pd  # type: ignore
            if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
                return str(value)
        except Exception:
            pass
        return str(value)

    for key, val in list(graph.graph.items()):
        graph.graph[key] = _coerce(val)
    for node, data in graph.nodes(data=True):
        for key, val in list(data.items()):
            data[key] = _coerce(val)
    for u, v, data in graph.edges(data=True):
        for key, val in list(data.items()):
            data[key] = _coerce(val)


def main() -> None:
    tickers = [t for t in config.tickers if t.strip()]
    start_date = config.start_date
    end_date = config.end_date

    cache = PriceCache()
    prices = cache.get_prices(tickers, start_date, end_date)
    company_names = cache.get_company_names(tickers)
    graph = _build_correlation_graph(prices)
    graph.graph["tickers"] = ",".join(tickers)
    graph.graph["start_date"] = start_date
    graph.graph["end_date"] = end_date
    for node in graph.nodes():
        graph.nodes[node]["name"] = company_names.get(node, node)
    _sanitize_for_graphml(graph)

    data_dir = Path("data") / "networks"
    data_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{start_date}-{end_date}-{_hash_tickers(tickers)}"
    graph_path = data_dir / f"{base_name}.graphml"
    meta_path = data_dir / f"{base_name}.json"

    nx.write_graphml(graph, graph_path)

    meta = {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "graph": graph_path.name,
        "names": [company_names.get(t, t) for t in tickers],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Correlation graph saved to: {graph_path}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
