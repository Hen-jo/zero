"""
논문 표 → PNG 생성 스크립트 (table1~4)
booktabs 스타일: 상하 굵은 선, 헤더 아래 얇은 선, 세로선 없음
출력: D:/paper/drzero/results/table{1-4}_*.png
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path("D:/paper/drzero/results")
OUT.mkdir(exist_ok=True)

LW_THICK = 1.4
LW_THIN  = 0.7
PAD      = 0.008


def _make_clip(ax, x0, y0, w, h):
    return patches.Rectangle((x0, y0), w, h, transform=ax.transData)


def _draw_cell(ax, text, x0, row_y0, cell_w, row_h, ha, fontsize,
               fontweight="normal", fontstyle="normal"):
    tx = (x0 + cell_w / 2 if ha == "center" else
          x0 + PAD        if ha == "left"   else
          x0 + cell_w - PAD)
    t = ax.text(tx, row_y0 + row_h / 2, text, ha=ha, va="center",
                fontsize=fontsize, fontweight=fontweight, fontstyle=fontstyle,
                clip_on=True)
    t.set_clip_path(_make_clip(ax, x0 + PAD, row_y0, cell_w - 2 * PAD, row_h))


def draw_table(
    ax,
    headers=None,            # simple list of header strings (or None for multilevel)
    rows=None,               # list of row lists
    col_widths=None,         # relative widths (auto-normalized)
    col_aligns=None,         # per-column alignment strings
    header_fontsize=9.5,
    cell_fontsize=9,
    row_height=0.13,
    bold_cols=None,          # column indices always bold
    bold_cells=None,         # [(r, c), ...] specific cells bold
    section_rows=None,       # {r_idx: label}  gray row inserted BEFORE data row r_idx
    mid_rules=None,          # [r_idx, ...]     thin line drawn AFTER data row r_idx
    multilevel_headers=None, # [[{text,start,span},...], ...] multi-level header rows
):
    # ── column setup ─────────────────────────────────────────────────────
    if headers:
        n_cols = len(headers)
    else:
        n_cols = max(d["start"] + d.get("span", 1)
                     for hrow in multilevel_headers for d in hrow)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    s = sum(col_widths)
    col_widths   = [w / s for w in col_widths]
    col_aligns   = col_aligns or ["center"] * n_cols
    bold_cols    = set(bold_cols or [])
    bold_cells   = set(map(tuple, bold_cells or []))
    section_rows = section_rows or {}
    mid_rules    = set(mid_rules or [])

    # ── build ordered visual list ─────────────────────────────────────────
    visual = []
    for r in range(len(rows)):
        if r in section_rows:
            visual.append(("sec", r, section_rows[r]))
        visual.append(("row", r))

    n_hdr   = len(multilevel_headers) if multilevel_headers else 1
    total_h = (n_hdr + len(visual)) * row_height

    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    def hline(y, lw):
        ax.axhline(y, xmin=0, xmax=1, linewidth=lw, color="black")

    hline(total_h, LW_THICK)

    # ── header row(s) ─────────────────────────────────────────────────────
    if multilevel_headers:
        for hi, hrow in enumerate(multilevel_headers):
            ry0 = total_h - (hi + 1) * row_height
            for d in hrow:
                x = sum(col_widths[: d["start"]])
                w = sum(col_widths[d["start"] : d["start"] + d.get("span", 1)])
                _draw_cell(ax, d["text"], x, ry0, w, row_height,
                           "center", header_fontsize, fontweight="bold")
            # draw underline beneath spanning group in first header row
            if hi == 0:
                for d in hrow:
                    if d.get("span", 1) > 1:
                        x = sum(col_widths[: d["start"]])
                        w = sum(col_widths[d["start"] : d["start"] + d["span"]])
                        ax.plot([x + 0.02, x + w - 0.02], [ry0, ry0],
                                color="black", linewidth=LW_THIN)
        hline(total_h - n_hdr * row_height, LW_THIN)
    else:
        ry0 = total_h - row_height
        x   = 0.0
        for h, w, a in zip(headers, col_widths, col_aligns):
            _draw_cell(ax, h, x, ry0, w, row_height, a,
                       header_fontsize, fontweight="bold")
            x += w
        hline(ry0, LW_THIN)

    # ── data rows ─────────────────────────────────────────────────────────
    cur_y = total_h - n_hdr * row_height
    for item in visual:
        ry0 = cur_y - row_height

        if item[0] == "sec":
            _, r_idx, label = item
            ax.add_patch(patches.Rectangle(
                (0, ry0), 1, row_height,
                transform=ax.transData, color="#eeeeee", zorder=0))
            ax.text(0.012, ry0 + row_height / 2, label,
                    ha="left", va="center", fontsize=cell_fontsize,
                    fontstyle="italic", color="#333333")
        else:
            _, r_idx = item
            x = 0.0
            for c, (cell, w, a) in enumerate(zip(rows[r_idx], col_widths, col_aligns)):
                fw = "bold" if (c in bold_cols or (r_idx, c) in bold_cells) else "normal"
                _draw_cell(ax, cell, x, ry0, w, row_height, a,
                           cell_fontsize, fontweight=fw)
                x += w
            if r_idx in mid_rules:
                hline(ry0, LW_THIN)

        cur_y = ry0

    hline(0, LW_THICK)


def save_fig(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Table 1 – Main results
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 3.8))
fig.patch.set_facecolor("white")
draw_table(
    ax,
    headers=["Method", "NQ", "TQA", "PopQA", "HQA", "2Wiki", "MSQ", "BGL", "Avg."],
    rows=[
        # Non-retrieval / supervised
        ["SFT",       "0.249", "0.292", "0.104", "0.186", "0.248", "0.044", "0.112", "0.176"],
        ["RAG",       "0.348", "0.544", "0.387", "0.255", "0.226", "0.047", "0.080", "0.270"],
        ["IRCoT",     "0.111", "0.312", "0.200", "0.164", "0.171", "0.067", "0.240", "0.181"],
        # Search-augmented RL
        ["Search-R1", "0.323", "0.537", "0.364", "0.308", "0.336", "0.105", "0.315", "0.327"],
        ["R-Zero",    "0.389", "0.513", "0.370", "0.243", "0.128", "0.052", "0.096", "0.256"],
        ["SQLM",      "0.264", "0.432", "0.258", "0.226", "0.238", "0.060", "0.158", "0.233"],
        # Self-play
        ["DrZero",    "0.397", "0.572", "0.431", "0.298", "0.291", "0.091", "0.200", "0.326"],
        ["Ours†",     "0.410", "0.584", "0.441", "0.304", "0.318", "0.105", "0.240", "0.343"],
    ],
    col_widths=[0.20] + [0.10] * 8,
    col_aligns=["left"] + ["center"] * 8,
    section_rows={0: "Non-retrieval / supervised", 3: "Search-augmented RL", 6: "Self-play"},
    bold_cells=[
        (7, 0), (7, 1), (7, 2), (7, 3),          # Ours†: method, NQ, TQA, PopQA
        (3, 4), (3, 5),                            # Search-R1: HQA, 2Wiki
        (3, 6), (7, 6),                            # MSQ tied best: Search-R1 & Ours†
        (3, 7),                                    # Search-R1: BGL
        (7, 8),                                    # Ours†: Avg.
    ],
    row_height=0.18,
)
save_fig(fig, "table1_main.png")


# ═════════════════════════════════════════════════════════════════════════════
# Table 2 – Response quality (LLM judge)
# Cols: Method(0) Iter(1) GPT-5.4(2) ClaudeS4.6(3) Gemini3.1Pro(4) Avg.(5) HNR%(6)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 2.4))
fig.patch.set_facecolor("white")
draw_table(
    ax,
    multilevel_headers=[
        [
            {"text": "Method",                   "start": 0, "span": 1},
            {"text": "Iter",                     "start": 1, "span": 1},
            {"text": "LLM-Valid (%, per judge)", "start": 2, "span": 3},
            {"text": "Avg.",                     "start": 5, "span": 1},
            {"text": "HNR (%)",                  "start": 6, "span": 1},
        ],
        [
            {"text": "GPT-5.4",        "start": 2, "span": 1},
            {"text": "Claude S4.6",    "start": 3, "span": 1},
            {"text": "Gemini 3.1 Pro", "start": 4, "span": 1},
        ],
    ],
    rows=[
        ["Base model", "1", "9.7",  "17.7", "18.2", "15.2", "54.2"],
        ["DrZero",     "2", "11.7", "18.3", "18.4", "16.1", "60.7"],
        ["",           "3", "10.5", "17.7", "18.7", "15.6", "58.9"],
        ["Ours",       "2", "18.8", "25.5", "27.7", "24.0", "68.3"],
        ["",           "3", "19.8", "26.4", "29.8", "25.3", "69.7"],
    ],
    col_widths=[0.18, 0.07, 0.15, 0.15, 0.17, 0.13, 0.15],
    col_aligns=["left", "center", "center", "center", "center", "center", "center"],
    mid_rules=[0, 2],
    bold_cells=[(4, 2), (4, 3), (4, 4), (4, 5), (4, 6)],
    row_height=0.20,
)
save_fig(fig, "table2_quality.png")


# ═════════════════════════════════════════════════════════════════════════════
# Table 3 – Ablation study
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 2.0))
fig.patch.set_facecolor("white")
draw_table(
    ax,
    headers=["Variant", "NQ", "TQA", "PopQA", "HQA", "2Wiki", "MSQ", "BGL", "Avg."],
    rows=[
        ["DrZero (base)",                "0.397", "0.572", "0.431", "0.298", "0.291", "0.091", "0.200", "0.326"],
        ["DrZero† (compute-matched)",    "0.388", "0.580", "0.425", "0.296", "0.294", "0.084", "0.177", "0.320"],
        ["Ours (real solver rollouts)",  "0.410", "0.584", "0.441", "0.304", "0.318", "0.105", "0.240", "0.343"],
        ["Ours (synthetic shuffled neg.)","0.411","0.577", "0.448", "0.298", "0.301", "0.094", "0.234", "0.338"],
    ],
    col_widths=[0.28] + [0.09] * 8,
    col_aligns=["left"] + ["center"] * 8,
    mid_rules=[1],
    bold_cells=[
        (3, 1),                               # NQ: 0.411
        (2, 2), (2, 4), (2, 5), (2, 6),      # TQA, HQA, 2Wiki, MSQ: Ours real
        (3, 3),                               # PopQA: 0.448
        (2, 7), (2, 8),                       # BGL, Avg.: Ours real
    ],
    row_height=0.19,
)
save_fig(fig, "table3_ablation.png")


# ═════════════════════════════════════════════════════════════════════════════
# Table 4 – Training dynamics across iterations
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 2.6))
fig.patch.set_facecolor("white")
draw_table(
    ax,
    headers=["Method", "Iter", "NQ", "TQA", "PopQA", "HQA", "2Wiki", "MSQ", "BGL", "Avg."],
    rows=[
        ["DrZero", "1", "0.381", "0.526", "0.392", "0.284", "0.243", "0.084", "0.216", "0.304"],
        ["",       "2", "0.401", "0.563", "0.408", "0.289", "0.255", "0.102", "0.216", "0.319"],
        ["",       "3", "0.397", "0.572", "0.431", "0.298", "0.291", "0.091", "0.200", "0.326"],
        ["Ours",   "1", "0.381", "0.526", "0.392", "0.284", "0.243", "0.084", "0.216", "0.304"],
        ["",       "2", "0.420", "0.600", "0.455", "0.315", "0.308", "0.098", "0.215", "0.344"],
        ["",       "3", "0.410", "0.584", "0.441", "0.304", "0.318", "0.105", "0.240", "0.343"],
    ],
    col_widths=[0.13, 0.07] + [0.10] * 8,
    col_aligns=["left", "center"] + ["center"] * 8,
    mid_rules=[2],
    bold_cells=[
        (4, 2), (4, 3), (4, 4), (4, 5), (4, 9),  # Ours iter2: NQ TQA PopQA HQA Avg
        (5, 6), (5, 7), (5, 8),                    # Ours iter3: 2Wiki MSQ BGL
    ],
    row_height=0.19,
)
save_fig(fig, "table4_dynamics.png")


print("\n완료: results/ 에 PNG 4개 저장됨")
