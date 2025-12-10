"""
Static correlation graph visualizer.

- Loads a GraphML file built by build_correlation_graph.py from data/networks/.
- Interactive slider for |corr| threshold and an MST toggle.
- Layout uses correlation-derived distances; saves a PNG on initial render.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import networkx as nx
import networkx.algorithms.community as nx_comm


def _load_metadata(path: Path) -> Dict:
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _list_graphs(data_dir: Path) -> List[Tuple[Path, Dict]]:
    paths = sorted(data_dir.rglob("*.graphml"))
    return [(p, _load_metadata(p)) for p in paths]


def _choose_entry(entries: List[Tuple[Path, Dict]]) -> Path:
    for idx, (path, meta) in enumerate(entries, start=1):
        tickers = meta.get("tickers") or []
        start_date = meta.get("start_date", "?")
        end_date = meta.get("end_date", "?")
        print(f"[{idx}] {path.name}  dates:{start_date}->{end_date}  tickers:{len(tickers)}")
    choice = input("Select a graph by number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(entries)):
        raise SystemExit("Invalid selection.")
    return entries[int(choice) - 1][0]


def _scale_widths(weights: List[float], min_width: float = 0.8, max_width: float = 6.0) -> List[float]:
    if not weights:
        return []
    abs_w = [abs(w) for w in weights]
    w_min, w_max = min(abs_w), max(abs_w)
    if w_max == w_min:
        return [0.5 * (min_width + max_width) for _ in abs_w]
    return [min_width + (max_width - min_width) * (w - w_min) / (w_max - w_min) for w in abs_w]


def _community_colors(graph: nx.Graph) -> Dict[str, str]:
    if graph.number_of_edges() == 0:
        return {n: "#111827" for n in graph.nodes()}
    for _, _, data in graph.edges(data=True):
        data["abs_corr"] = abs(float(data.get("corr", data.get("weight", 0.0))))
    communities = list(nx_comm.greedy_modularity_communities(graph, weight="abs_corr"))
    palette = [
        "#0ea5e9",
        "#f97316",
        "#22c55e",
        "#a855f7",
        "#ef4444",
        "#14b8a6",
        "#eab308",
        "#6366f1",
    ]
    colors = {}
    for idx, comm in enumerate(communities):
        color = palette[idx % len(palette)]
        for node in comm:
            colors[node] = color
    for n in graph.nodes():
        colors.setdefault(n, "#111827")
    return colors


def _prepare_graph(graph_path: Path) -> nx.Graph:
    graph = nx.read_graphml(graph_path)
    for node in graph.nodes():
        graph.nodes[node]["name"] = graph.nodes[node].get("name", node)
    for u, v, data in graph.edges(data=True):
        weight_raw = data.get("weight", 0.0)
        try:
            weight_val = float(weight_raw)
        except (TypeError, ValueError):
            weight_val = 0.0
        data["weight"] = weight_val
        data["corr"] = weight_val
        data["abs_corr"] = abs(weight_val)
        clipped = max(-1.0, min(1.0, weight_val))
        data["distance"] = (2 * (1 - clipped)) ** 0.5
        data["sign"] = 1 if weight_val >= 0 else -1
    return graph


def _filter_graph(graph: nx.Graph, threshold: float) -> nx.Graph:
    filtered = graph.copy()
    filtered.remove_edges_from(
        [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if abs(float(data.get("corr", data.get("weight", 0.0)))) < threshold
        ]
    )
    filtered.remove_nodes_from(list(nx.isolates(filtered)))
    return filtered


def _minimum_spanning_tree(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_edges() == 0:
        return graph.copy()
    return nx.minimum_spanning_tree(graph, weight="distance")


def _layout_positions(graph: nx.Graph) -> Dict:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.kamada_kawai_layout(graph, weight="distance")
    except Exception:
        return nx.spring_layout(graph, seed=42, weight="distance")


def _interactive_plot(graph_path: Path, meta: Dict) -> None:
    graph = _prepare_graph(graph_path)
    if graph.number_of_nodes() == 0:
        raise SystemExit("Graph is empty; cannot visualize.")

    title_base = f"Correlation Graph ({meta.get('start_date', '?')} to {meta.get('end_date', '?')})"

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.22)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    slider_ax = fig.add_axes([0.15, 0.10, 0.7, 0.04])
    button_ax = fig.add_axes([0.15, 0.02, 0.2, 0.05])

    threshold_slider = widgets.Slider(
        ax=slider_ax,
        label="Min |corr|",
        valmin=0.0,
        valmax=1.0,
        valinit=0.5,
        valstep=0.01,
    )
    mst_mode = {"enabled": False}
    mst_button = widgets.Button(button_ax, "Toggle MST (off)")

    output_path = graph_path.with_suffix(".png")
    base_layout = _layout_positions(graph)

    def draw(threshold: float, save: bool = False) -> None:
        ax.clear()
        filtered = _filter_graph(graph, threshold)
        plot_graph = _minimum_spanning_tree(filtered) if mst_mode["enabled"] else filtered

        node_colors = _community_colors(plot_graph)
        raw_weights = [float(d.get("corr", 0.0)) for _, _, d in plot_graph.edges(data=True)]
        colors = ["#1f77b4" if w >= 0 else "#d62728" for w in raw_weights]
        widths = _scale_widths(raw_weights)
        degrees = dict(plot_graph.degree())
        node_sizes = [200 + 40 * degrees.get(n, 0) for n in plot_graph.nodes()]
        labels = {n: plot_graph.nodes[n].get("name", n) for n in plot_graph.nodes()}

        pos = {n: base_layout[n] for n in plot_graph.nodes() if n in base_layout}
        if len(pos) != plot_graph.number_of_nodes():
            pos = _layout_positions(plot_graph)

        nx.draw_networkx_nodes(
            plot_graph,
            pos,
            node_color=[node_colors.get(n, "#111827") for n in plot_graph.nodes()],
            node_size=node_sizes,
            alpha=0.9,
            ax=ax,
        )
        nx.draw_networkx_labels(plot_graph, pos, labels=labels, font_size=8, font_color="black", ax=ax)
        nx.draw_networkx_edges(
            plot_graph,
            pos,
            edge_color=colors if colors else "#9ca3af",
            width=widths if widths else 1,
            alpha=0.8,
            ax=ax,
        )
        kept_edges = plot_graph.number_of_edges()
        total_edges = graph.number_of_edges()
        ax.set_title(
            f"{title_base}"
            + (" [MST]" if mst_mode['enabled'] else "")
            + f"\n|corr| ≥ {threshold:.2f} | edges: {kept_edges}/{total_edges}"
        )
        ax.axis("off")
        fig.tight_layout()
        if save:
            fig.savefig(output_path, dpi=200)
        fig.canvas.draw_idle()

    draw(threshold_slider.val, save=True)

    def on_threshold(val: float) -> None:
        draw(val, save=False)

    def on_toggle(event) -> None:
        mst_mode["enabled"] = not mst_mode["enabled"]
        mst_button.label.set_text("Toggle MST (on)" if mst_mode["enabled"] else "Toggle MST (off)")
        draw(threshold_slider.val, save=False)

    threshold_slider.on_changed(on_threshold)
    mst_button.on_clicked(on_toggle)

    plt.show()
    plt.close(fig)


def main() -> None:
    data_dir = Path("data") / "networks"
    entries = _list_graphs(data_dir)
    if not entries:
        raise SystemExit("No graph files found in data/networks/ (expected *.graphml).")

    graph_path = _choose_entry(entries)
    meta = _load_metadata(graph_path)
    _interactive_plot(graph_path, meta)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
