import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import igraph as ig
import numpy as np
import seaborn as sns
import json

import collections
import xnetwork as xn
import powerlaw
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import pandas as pd

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors  # for LogNorm

import pandas as pd
import networkx as nx
import random
import numpy as np

# Tente usar SciPy para Kendall tau; se não houver, seguimos sem Kendall.
try:
    from scipy.stats import kendalltau as _kendalltau
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    def _kendalltau(a, b):
        raise ImportError("SciPy não disponível: kendalltau indisponível.")


def plot_by_paper(GG, paper='1801.07698'):
    # Assume GG is an existing graph with node attributes (including 'label')
    # Create directed graph G using only the in-edges for '1801.07698'
    G = nx.DiGraph()
    G.add_edges_from(GG.in_edges(paper))
    
    # Use a spring layout with a fixed seed for reproducibility
    pos = nx.spring_layout(G, seed=42)
    
    # Extract node labels from GG (assuming the 'label' attribute exists)
    node_labels_attr = nx.get_node_attributes(GG, 'label')
    
    # Filter the labels to only include nodes present in G
    used_labels = {node_labels_attr[node] for node in G.nodes() if node in node_labels_attr}
    
    # Define a custom color palette (using hex codes)
    color_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
                     '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    
    # Create a mapping from label to color (sorted for consistency)
    label_color_map = {label: color_palette[i % len(color_palette)] 
                       for i, label in enumerate(sorted(used_labels))}
    
    # Build the node_colors list based on the label for each node in G:
    # - '1801.07698' is highlighted in black.
    # - Nodes with a label are colored according to label_color_map.
    # - Nodes without a label are given a default light gray color.
    node_colors = []
    for node in G.nodes():
        if node == paper:
            node_colors.append('#000000')  # Black for the central paper
        elif node in node_labels_attr:
            node_colors.append(label_color_map[node_labels_attr[node]])
        else:
            node_colors.append('#cccccc')  # Default color for unlabeled nodes
    
    # Create the figure for a publication-quality plot
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, edge_color='#888888', alpha=0.8)
    # nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.title(f"Network of Papers Connected to {paper}", fontsize=16)
    plt.axis('off')
    
    # Build legend only for the labels that appear in G
    legend_elements = []
    for label, color in label_color_map.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label,
                                      markerfacecolor=color, markersize=10))
    # Optionally, add an entry for the central paper
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=paper,
                                  markerfacecolor='#000000', markersize=10))
    plt.legend(handles=legend_elements, loc='upper left', title="Labels", fontsize=10, title_fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure as a PDF for inclusion in your paper
    pdf_filename = "network_graph.pdf"
    plt.savefig(pdf_filename, format='pdf')
    plt.show()


def plot_top_cited_ids(
    df: pd.DataFrame,
    column: str = "cited_ids",
    top_n: int = 20,
    title: str = "Top Cited IDs",
    figsize: tuple = (10, 6),
    annotate_offset: float = 0.01
):
    # compute counts
    counts = df[column].value_counts().iloc[:top_n]

    # create plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color='gray',
        edgecolor='black'
    )

    # clean up spines & add grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # labels and title
    ax.set_xlabel(column.replace('_', ' ').title(), fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='x', labelrotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # annotate bars (inside)
    max_h = counts.values.max()
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max_h * annotate_offset,
            f'{int(h):,}',
            ha='center',
            va='bottom',
            fontsize=13
        )

    plt.tight_layout()
    pdf_filename = "top_20_cited_papers.pdf"
    plt.savefig(pdf_filename, format='pdf')
    plt.show()



def plot_value_counts_by_group(
    df: pd.DataFrame,
    column: str,
    bins: list[float],
    labels: list[str],
    title: str = None,
    figsize: tuple = (8, 5),
    annotate_offset: float = 0.01
):
    # 1) value_counts → DataFrame
    s = df[column].value_counts()
    df_counts = (
        s.rename_axis(column)
         .reset_index(name='freq')
    )

    # 2) bin the unique values
    df_counts['group'] = pd.cut(
        df_counts[column],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    # 3) sum frequencies per group, keep original order
    grouped = df_counts.groupby('group')['freq'].sum().reindex(labels)

    # 4) plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        grouped.index,
        grouped.values,
        color='gray',
        edgecolor='black'
    )

    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Labels and title
    ax.set_xlabel(f'{column} group', fontsize=16)
    ax.set_ylabel('Total number of records', fontsize=16)
    ax.set_title(title or f'Distribution of {column} by Group', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    plt.xticks(rotation=45, ha='right')

    # 5) annotate bars
    max_h = grouped.values.max()
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max_h * annotate_offset,
            f'{int(h):,}',
            ha='center',
            va='bottom',
            fontsize=13
        )

    plt.tight_layout()
    plt.show()


def _series_from_mapping(m: Dict[str, float]) -> pd.Series:
    """Converte dict {id: valor} em Series indexada por id (float)."""
    s = pd.Series(m, dtype=float)
    # Garantir ordenação consistente de índice para empates/empates no top-k
    s = s.sort_index()
    return s


def _ranks_from_series(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Converte valores em ranks (1 = melhor).
    Usa metodo 'average' para empates (apropriado para Spearman).
    """
    return s.rank(ascending=not higher_is_better, method="average")


def _align_ranks(base: pd.Series, other: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Alinha por interseção de IDs e retorna ranks alinhados (mesmo índice).
    """
    ids = base.index.intersection(other.index)
    return base.loc[ids], other.loc[ids]


def _spearman_from_ranks(r0: pd.Series, r1: pd.Series) -> float:
    """Correlação de Spearman a partir de ranks (sem SciPy)."""
    x = r0.to_numpy(dtype=float)
    y = r1.to_numpy(dtype=float)
    # Spearman = Pearson(ranks)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _kendall_tau(r0: pd.Series, r1: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Kendall tau (τ) e p-valor (se SciPy disponível)."""
    if not _HAVE_SCIPY:
        return (None, None)
    tau, pval = _kendalltau(r0.to_numpy(dtype=float), r1.to_numpy(dtype=float))
    return (float(tau), float(pval))


def _topk_ids(s: pd.Series, k: int) -> List[str]:
    """IDs do top-k por valor (decrescente), desempate por ID (índice)."""
    if k <= 0:
        return []
    # Ordena por (-valor, id) usando sort_values com kind estável + sort_index já feito
    ordered = s.sort_values(ascending=False, kind="mergesort")
    return ordered.index[:min(k, len(ordered))].tolist()


def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def _delta_rank_stats(r0: pd.Series, r1: pd.Series, thresholds: List[int]) -> Dict[str, float]:
    """Resumo estatístico de Δr."""
    d = (r1 - r0).to_numpy(dtype=float)
    out = {
        "n": len(d),
        "delta_mean": float(np.mean(d)) if len(d) else np.nan,
        "delta_median": float(np.median(d)) if len(d) else np.nan,
        "delta_std": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
        "p10": float(np.percentile(d, 10)) if len(d) else np.nan,
        "p90": float(np.percentile(d, 90)) if len(d) else np.nan,
        "p95": float(np.percentile(d, 95)) if len(d) else np.nan,
        "max_drop": float(np.max(d)) if len(d) else np.nan,   # piorou (rank maior)
        "max_gain": float(np.min(d)) if len(d) else np.nan,   # melhorou (rank menor)
    }
    # Frações acima de limiares de magnitude
    absd = np.abs(d)
    for T in thresholds:
        if len(d):
            out[f"frac_|Δr|>{T}"] = float(np.mean(absd > T))
        else:
            out[f"frac_|Δr|>{T}"] = np.nan
    return out


def _topk_churn(base: pd.Series, other: pd.Series, k: int) -> Dict[str, float]:
    A = set(_topk_ids(base, k))
    B = set(_topk_ids(other, k))
    if k == 0:
        return {"k": 0, "enter": 0.0, "leave": 0.0, "stable": 1.0}
    enter = len(B - A) / k
    leave = len(A - B) / k
    stable = len(A & B) / k
    return {"k": k, "enter": enter, "leave": leave, "stable": stable}


def _pct_to_k(n: int, pct: float) -> int:
    k = int(round(pct * n))
    return max(1, min(k, n))


def compare_scenario(
    data: Dict[str, Dict[str, Dict[str, float]]],
    baseline_key: str,
    scenario_key: str,
    measure: str,
    topk_list: List[int] = [10, 50, 100, 500],
    pct_list: List[float] = [0.01, 0.05],
    thresholds: List[int] = [10, 50, 100, 500],
) -> Dict[str, any]:
    base_s = _series_from_mapping(data[baseline_key][measure])
    oth_s  = _series_from_mapping(data[scenario_key][measure])

    # ranks (1 = melhor)
    r0 = _ranks_from_series(base_s, higher_is_better=True)
    r1 = _ranks_from_series(oth_s,  higher_is_better=True)

    # alinhar
    r0a, r1a = _align_ranks(r0, r1)
    n = len(r0a)

    # correlações
    rho = _spearman_from_ranks(r0a, r1a)
    tau, pval = _kendall_tau(r0a, r1a)

    # Δr estatísticas
    delta_stats = _delta_rank_stats(r0a, r1a, thresholds)

    # Jaccard top-k (usar os mesmos k mas limitados por n)
    jacc = {}
    for k in topk_list:
        kk = min(k, n)
        A = _topk_ids(base_s.loc[r0a.index], kk)
        B = _topk_ids(oth_s.loc[r1a.index],  kk)
        jacc[f"J@{kk}"] = _jaccard(A, B)

    # churn em percentis (top 1%, 5% por padrão)
    churn_rows = []
    for pct in pct_list:
        kk = _pct_to_k(n, pct)
        churn_rows.append({"pct": pct, **_topk_churn(base_s.loc[r0a.index], oth_s.loc[r1a.index], kk)})

    return {
        "scenario": scenario_key,
        "measure": measure,
        "n_common": n,
        "spearman": rho,
        "kendall_tau": tau,
        "kendall_p": pval,
        "delta": delta_stats,       # dict
        "jaccard": jacc,            # dict
        "churn": churn_rows,        # list[dict]
    }


def evaluate_all(
    data: Dict[str, Dict[str, Dict[str, float]]],
    baseline_key: str = "Full Network",
    scenarios: List[str] = ("Background Filtered", "Method Filtered", "Result Filtered"),
    measures: List[str] = ("degree_in", "degree_out", "pagerank", "betweenness", "closeness", "eigenvector"),
    topk_list: List[int] = [10, 50, 100, 500],
    pct_list: List[float] = [0.01, 0.05],
    thresholds: List[int] = [10, 50, 100, 500],
) -> Dict[str, pd.DataFrame]:
    rows_corr, rows_delta, rows_jacc, rows_churn = [], [], [], []

    for scen in scenarios:
        for m in measures:
            res = compare_scenario(
                data,
                baseline_key=baseline_key,
                scenario_key=scen,
                measure=m,
                topk_list=topk_list,
                pct_list=pct_list,
                thresholds=thresholds,
            )

            rows_corr.append({
                "scenario": res["scenario"],
                "measure": res["measure"],
                "n_common": res["n_common"],
                "spearman": res["spearman"],
                "kendall_tau": res["kendall_tau"],
                "kendall_p": res["kendall_p"],
            })

            rows_delta.append({
                "scenario": res["scenario"],
                "measure": res["measure"],
                **res["delta"]
            })

            jrow = {"scenario": res["scenario"], "measure": res["measure"]}
            jrow.update(res["jaccard"])
            rows_jacc.append(jrow)

            for c in res["churn"]:
                rows_churn.append({
                    "scenario": res["scenario"],
                    "measure": res["measure"],
                    "pct": c["pct"],
                    "k": c["k"],
                    "enter": c["enter"],
                    "leave": c["leave"],
                    "stable": c["stable"],
                })

    correlations_df = pd.DataFrame(rows_corr)
    deltas_df       = pd.DataFrame(rows_delta)
    jaccard_df      = pd.DataFrame(rows_jacc)
    churn_df        = pd.DataFrame(rows_churn)

    # Ordena para leitura
    sort_cols = ["scenario", "measure"]
    correlations_df = correlations_df.sort_values(sort_cols).reset_index(drop=True)
    deltas_df       = deltas_df.sort_values(sort_cols).reset_index(drop=True)
    jaccard_df      = jaccard_df.sort_values(sort_cols).reset_index(drop=True)
    churn_df        = churn_df.sort_values(sort_cols + ["pct"]).reset_index(drop=True)

    return {
        "correlations_df": correlations_df,
        "deltas_df": deltas_df,
        "jaccard_df": jaccard_df,
        "churn_df": churn_df,
    }

def plot_rank_histograms_by_intent(
    centralities,
    output_dir,
    intents=None,
    bins=50,
    cmap="Blues",
    invert=False,
    log_axes=False,             # use log scale on x and y axes
    log_color=False,            # use log scale for color (counts)
    top_k=None,                 # only show top-k ranks (if not None)
    text_width=None,
    text_mode="wrap",
    title_fs=14,
    label_fs=11,
    tick_fs=10,
    tick_scale_exp="auto",      # int (e.g., 3 -> ×10^3), or "auto"
    max_tick_labels=9,          # when auto, keep labeled ticks ≤ this
    hide_tick=False,
    rank_tick_step=None
):
    import textwrap

    # method = "dense"
    method = "first"
    def _format_text(s: str) -> str:
        if text_width is None:
            return s
        if text_mode == "wrap":
            return textwrap.fill(s, width=max(1, int(text_width)))
        w = max(1, int(text_width))
        return s if len(s) <= w else s[: max(1, w - 1)] + "…"

    def _auto_exp(n: int, max_labels: int) -> int:
        if n <= 0:
            return 0
        if n <= max_labels:
            return 0
        # choose largest k such that floor(n/10^k) is within [1, max_labels]
        k_max = int(np.floor(np.log10(n)))
        for k in range(k_max, -1, -1):
            m = int(np.floor(n / (10**k)))
            if 1 <= m <= max_labels:
                return k
        return 0

    if intents is None:
        intents = {
            "Background": "Background Filtered",
            "Method":     "Method Filtered",
            "Result":     "Result Filtered",
        }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    full_net = centralities["Full Network"]

    for measure, full_scores in full_net.items():
        # Base dataframe
        papers = list(full_scores.keys())
        df = pd.DataFrame({
            "paper": papers,
            "full":  [full_scores[p] for p in papers],
        })
        df["rank_full"] = df["full"].rank(ascending=False, method=method).astype(int)
        # print(df.head(50))
        # break
        n_all = int(df["rank_full"].max())

        # Restrict to top_k if requested
        if top_k is None:
            max_rank = n_all
        else:
            max_rank = min(int(top_k), n_all)

        rank_range = [[1, max_rank], [1, max_rank]]

        if not log_axes:
            # NEW: explicit tick step in *rank units* (1, 10, 20, ..., N)
            if rank_tick_step is not None:
                step = int(rank_tick_step)
                if step <= 0:
                    raise ValueError("rank_tick_step must be a positive integer")

                tick_pos = np.unique(
                    np.clip(
                        np.r_[1, np.arange(step, max_rank + 1, step)],
                        1, max_rank
                    )
                ).astype(int)
                tick_lbl = [str(t) for t in tick_pos]
                tick_scale_info = 0  # no scaling info
            else:
                # your current scaled ticks using tick_scale_exp
                if tick_scale_exp == "auto":
                    exp = _auto_exp(max_rank, max_tick_labels)
                else:
                    exp = int(tick_scale_exp)
                s = 10 ** max(0, exp)
                m = max(1, int(np.floor(max_rank / s)))  # number of labeled ticks (1..m)

                tick_pos = (np.arange(1, m + 1) * s).astype(int)   # positions in rank space
                tick_lbl = [str(j) for j in range(1, m + 1)]       # labels 1..m
                tick_scale_info = exp
        else:
            # Log axes: ticks at powers of 10 within [1, max_rank]
            if max_rank < 1:
                tick_pos = np.array([1])
                tick_lbl = ["1"]
                tick_scale_info = 0
            else:
                exp_min = 0
                exp_max = int(np.floor(np.log10(max_rank)))
                exps = np.arange(exp_min, exp_max + 1)
                tick_pos = (10 ** exps).astype(int)
                tick_lbl = [f"$10^{e}$" if e > 0 else "1" for e in exps]
                tick_scale_info = None

        # --- Figure layout: 3 panels + dedicated colorbar column ---
        fig = plt.figure(figsize=(18, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cax  = fig.add_subplot(gs[0, 3])

        # --- First pass: global vmax for color scale ---
        vmax = 0
        cache = []
        for intent_name, filt_key in intents.items():
            filt_scores = centralities[filt_key][measure]
            df[f"filtered_{intent_name}"] = [filt_scores.get(p, 0) for p in df["paper"]]
            df[f"rank_filtered_{intent_name}"] = (
                df[f"filtered_{intent_name}"]
                .rank(ascending=False, method=method)
                .astype(int)
            )

            x = df["rank_full"].to_numpy()
            y = df[f"rank_filtered_{intent_name}"].to_numpy()

            # Apply top_k restriction
            if top_k is not None:
                mask = (x <= max_rank) & (y <= max_rank)
                x = x[mask]
                y = y[mask]

            H, _, _ = np.histogram2d(x, y, bins=bins, range=rank_range)
            if H.size > 0:
                vmax = max(vmax, H.max())
            cache.append((intent_name, x, y))

        if vmax <= 0:
            vmax = 1  # avoid degenerate normalization

        # --- Color normalization: linear vs log ---
        if log_color:
            # vmin=1 (ignore empty bins); everything between 1 and vmax on log scale
            norm = colors.LogNorm(vmin=1, vmax=vmax)
        else:
            norm = None

        # --- Plot panels ---
        quadmesh = None
        for ax, (intent_name, x, y) in zip(axes, cache):
            hist_kwargs = dict(
                x=x,
                y=y,
                bins=bins,
                range=rank_range,
                cmap=cmap,
            )
            if log_color:
                # Use norm only (no vmin/vmax) to avoid ValueError
                hist_kwargs["norm"] = norm
            else:
                hist_kwargs["vmin"] = 0
                hist_kwargs["vmax"] = vmax

            h = ax.hist2d(**hist_kwargs)
            quadmesh = h[3]

            # Identity line
            ax.plot([1, max_rank], [1, max_rank],
                    linestyle="--", linewidth=2.0, color="red")

            ax.set_title(_format_text(f"No {intent_name}"), fontsize=title_fs)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)

            # Limits
            ax.set_xlim(1, max_rank)
            ax.set_ylim(1, max_rank)

            # Axis scale: log or linear
            if log_axes:
                ax.set_xscale("log")
                ax.set_yscale("log")

            if invert:
                ax.invert_xaxis()
                ax.invert_yaxis()

            # Ticks
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, fontsize=tick_fs)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_lbl, fontsize=tick_fs)
            ax.set_facecolor('black')

        # Hide y tick labels on 2nd & 3rd panels if requested
        if hide_tick:
            for ax in axes[1:]:
                ax.set_yticklabels([])

        # --- Axis labels ---
        # If we used scaled linear ticks, mention ×10^exp; else just plain label.
        if not log_axes and tick_scale_info and tick_scale_info > 0:
            scale_str = f" (×10^{tick_scale_info})"
        else:
            scale_str = ""

        axes[0].set_ylabel(
            _format_text("Rank after filtering" + scale_str),
            fontsize=label_fs
        )
        for ax in axes:
            ax.set_xlabel(
                _format_text("Rank before filtering" + scale_str),
                fontsize=label_fs
            )

        # --- Shared colorbar ---
        cbar = fig.colorbar(quadmesh, cax=cax)
        cbar_label = "Number of papers"
        if log_color:
            cbar_label += " (log scale)"
        cbar.set_label(_format_text(cbar_label), fontsize=label_fs)
        cbar.ax.tick_params(labelsize=tick_fs)

        # --- Title & save ---
        suffix_parts = []
        if log_axes:
            suffix_parts.append("log–log axes")
        # if log_color:
        #     suffix_parts.append("log color")
        if top_k is not None:
            suffix_parts.append(f"top {max_rank}")

        if suffix_parts:
            extra = " (" + ", ".join(suffix_parts) + ")"
        else:
            extra = ""

        fig.suptitle(
            _format_text(f"{measure.capitalize()} — Rank changes by removed intent{extra}"),
            y=1.02, fontsize=title_fs
        )
        fig.subplots_adjust(top=0.90)

        # Filename with suffixes
        base_name = f"{measure.lower()}_rank_changes_by_intent"
        if log_axes:
            base_name += "_logaxes"
        if log_color:
            base_name += "_logcolor"
        if top_k is not None:
            base_name += f"_top{max_rank}"

        out_path = Path(output_dir) / f"{base_name}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        # plt.show()
        plt.close(fig)

    return df

def build_rank_dictionaries_safe(centralities, measure="closeness"):

    # Extract full network scores
    full_scores = centralities["Full Network"][measure]

    df_full = pd.DataFrame({
        "paper": list(full_scores.keys()),
        "full":  list(full_scores.values()),
    })

    # Compute full-network ranks
    df_full["rank_full"] = df_full["full"].rank(
        ascending=False, method="first"
    ).astype(int)

    rank_full_dictionary = dict(zip(df_full["paper"], df_full["rank_full"]))

    # Intent mappings
    intents = {
        "Background": "Background Filtered",
        "Method":     "Method Filtered",
        "Result":     "Result Filtered",
    }

    rank_filtered_dictionary = {}

    for intent_name, filt_key in intents.items():

        filt_scores = centralities[filt_key][measure]

        # Intersection of papers present in both full and filtered networks
        common_papers = set(full_scores.keys()) & set(filt_scores.keys())

        df_filt = pd.DataFrame({
            "paper": list(common_papers),
            "filtered": [filt_scores[p] for p in common_papers],
        })

        # Compute ranks within the filtered network
        df_filt["rank_filtered"] = df_filt["filtered"].rank(
            ascending=False, method="first"
        ).astype(int)

        # Store dictionary for this intent
        rank_filtered_dictionary[intent_name] = dict(
            zip(df_filt["paper"], df_filt["rank_filtered"])
        )

    return rank_full_dictionary, rank_filtered_dictionary


def find_closeness_plateaus(centralities, filt_key="Background Filtered",
                            measure="closeness", min_size=50):
    scores = centralities[filt_key][measure]  # dict: paper -> closeness

    df = pd.DataFrame({
        "paper": list(scores.keys()),
        "closeness": list(scores.values()),
    })

    grouped = df.groupby("closeness")["paper"].apply(list)
    plateaus = [(val, papers) for val, papers in grouped.items()
                if len(papers) >= min_size]

    # sort by plateau size (largest first)
    plateaus.sort(key=lambda x: len(x[1]), reverse=True)
    return plateaus

def analyze_plateau_module(G, plateau_nodes, n_random=20):
    plateau_nodes = [n for n in plateau_nodes if n in G]  # intersect with graph nodes
    k = len(plateau_nodes)
    if k == 0:
        raise ValueError("No plateau nodes present in graph.")

    sub = G.subgraph(plateau_nodes)

    # basic stats for plateau subgraph
    density_plateau = nx.density(sub)
    n_comp = nx.number_connected_components(sub)
    avg_deg = sum(dict(sub.degree()).values()) / k

    # random baseline: same size sets
    densities_rand = []
    avg_deg_rand = []
    nodes_list = list(G.nodes())

    for _ in range(n_random):
        sample_nodes = random.sample(nodes_list, k)
        sub_r = G.subgraph(sample_nodes)
        densities_rand.append(nx.density(sub_r))
        if len(sub_r) > 0:
            avg_deg_rand.append(sum(dict(sub_r.degree()).values()) / max(1, len(sub_r)))
        else:
            avg_deg_rand.append(0.0)

    result = {
        "k": k,
        "density_plateau": density_plateau,
        "density_random_mean": float(np.mean(densities_rand)),
        "density_random_std": float(np.std(densities_rand)),
        "avg_deg_plateau": avg_deg,
        "avg_deg_random_mean": float(np.mean(avg_deg_rand)),
        "avg_deg_random_std": float(np.std(avg_deg_rand)),
        "n_components_plateau": n_comp,
    }
    return result

def find_method_worseners(rank_full, rank_method, quantile=0.99):
    """
    Returns the list of papers whose rank worsened the most when Method sentences were removed.
    """
    rows = []
    for p in rank_method.keys():
        if p in rank_full:
            rf = rank_full[p]
            rm = rank_method[p]
            jump = rm - rf   # positive means worse
            rows.append((p, rf, rm, jump))

    df = pd.DataFrame(rows, columns=["paper","rank_full","rank_method","jump"])

    # restrict to the biggest worsening (top 1% by default)
    thresh = df["jump"].quantile(quantile)
    df_worst = df[df["jump"] >= thresh].sort_values("jump", ascending=False)

    return df, df_worst

def compute_lost_edges(G_full, G_method, papers):
    lost = []
    for p in papers:
        if p in G_full and p in G_method:
            deg_full = G_full.degree(p)
            deg_method = G_method.degree(p)
            lost_edges = deg_full - deg_method
            lost.append((p, deg_full, deg_method, lost_edges))
    return pd.DataFrame(lost, columns=["paper","deg_full","deg_method","lost_edges"])

def compute_lost_edges_with_neighbors(G_full, G_filtered, papers):
    records = []

    for p in papers:
        if p not in G_full or p not in G_filtered:
            continue

        # neighbors in full graph
        neigh_full = set(G_full.neighbors(p))

        # neighbors in method-filtered graph
        neigh_filtered = set(G_filtered.neighbors(p))

        # edges that existed but vanished after filtering
        lost_neighbors = list(neigh_full - neigh_filtered)

        records.append({
            "paper": p,
            "deg_full": len(neigh_full),
            "deg_filtered": len(neigh_filtered),
            "lost_edges": len(lost_neighbors),
            "lost_neighbors": lost_neighbors  # <-- list of papers
        })

    return pd.DataFrame(records)