#!/usr/bin/env python3
"""Draw the figures that go with docs/mxfp4_ablation_report.md.

Reads the same summary.json files summarize.py writes, plus the per-arm autotune
caches and raw logs for the two figures that need per-iteration detail.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PLOT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "results" / "ablation"
DOC_CHARTS = pathlib.Path(__file__).resolve().parents[3] / "docs" / "charts"

# A6 is the one real slowdown, A18 the one that buys speed with memory, A24 only
# shows up in the tail. Everything else is coloured by whether its rung cleared
# the noise floor, which a single run puts at roughly 1%.
SLOWER, MEMTRADE, TAIL = 6, 18, 24
NOISE_PCT = 1.0

COLORS = {
    "gain": "#2f7d5c",
    "noise": "#b9bcc0",
    "slower": "#c1443c",
    "memtrade": "#dd8b2a",
    "tail": "#4a72b0",
    "base": "#6b7075",
}
LEGEND = [
    ("gain", "无损，越过噪声线"),
    ("noise", "落在噪声内（±1%）"),
    ("slower", "变慢"),
    ("memtrade", "换显存"),
    ("tail", "只动尾延迟"),
]

GROUPS = [
    ("backward\n算子链", [1, 2, 5, 11, 14, 15, 16, 17, 18, 19, 20], "#2f7d5c"),
    ("非 MXFP4\n共享 kernel", [13, 21, 22, 23], "#4a72b0"),
    ("MXFP4 GEMM\nbackend / tuning", [3, 6, 7, 8, 9, 10, 12], "#dd8b2a"),
    ("显存 / GC", [4, 24], "#b9bcc0"),
]

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
)


def use_cjk_font() -> None:
    """Pick a font that covers both the Chinese labels and the ASCII numbers.

    matplotlib does not fall back per glyph here, so a CJK-only font (the
    preinstalled Droid Sans Fallback, for one) drops every digit instead. Fail
    loudly rather than write a figure full of empty boxes.
    """
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans SC", "Noto Sans CJK SC", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "Microsoft YaHei"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    raise SystemExit(
        "no CJK font with Latin coverage found. Install one, e.g.\n"
        "  curl -fsSL -o ~/.local/share/fonts/NotoSansSC-Regular.ttf \\\n"
        "    'https://raw.githubusercontent.com/google/fonts/main/ofl/"
        "notosanssc/NotoSansSC%5Bwght%5D.ttf' && fc-cache -f\n"
        "then delete ~/.cache/matplotlib so it rescans."
    )


def style() -> None:
    use_cjk_font()
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c8ccd0",
        "axes.grid": True,
        "grid.color": "#e8eaec",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


class Ladder:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.arms: list[int] = []
        self.summary: dict[int, dict] = {}
        self.label: dict[int, str] = {}
        for d in sorted(root.glob("S*"), key=lambda p: int(p.name[1:])):
            f = d / "run1" / "summary.json"
            if not f.is_file():
                continue
            n = int(d.name[1:])
            self.arms.append(n)
            self.summary[n] = json.loads(f.read_text())
            tag = d / "run1" / "arm.txt"
            self.label[n] = tag.read_text().strip().lstrip("+ ") if tag.is_file() else d.name

    def median(self, n: int) -> float:
        return self.summary[n]["median_ms"]

    def step_pct(self, n: int) -> float:
        """Per-rung change, negative = faster."""
        return (self.median(n) / self.median(n - 1) - 1) * 100

    def category(self, n: int) -> str:
        if n == SLOWER:
            return "slower"
        if n == MEMTRADE:
            return "memtrade"
        if n == TAIL:
            return "tail"
        return "noise" if abs(self.step_pct(n)) <= NOISE_PCT else "gain"

    def iters(self, n: int, warmup: int = 15) -> list[tuple[int, float]]:
        log = self.root / f"S{n}" / "run1" / "train.log"
        got = [(int(m.group(1)), float(m.group(2)))
               for m in ITER_RE.finditer(log.read_text())]
        return [(i, v) for i, v in got if i > warmup]

    def backends(self, n: int) -> collections.Counter:
        f = self.root / f"S{n}" / "run1" / "autotune.json"
        if not f.is_file():
            return collections.Counter()
        return collections.Counter(json.loads(f.read_text())["choices"].values())


def fig_staircase(L: Ladder, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    xs = L.arms
    ys = [L.median(n) for n in xs]
    cats = ["base"] + [L.category(n) for n in xs[1:]]

    ax.bar(xs, ys, color=[COLORS[c] for c in cats], width=0.74, zorder=3)
    ax.plot(xs, ys, color="#2b2f33", lw=1.4, marker="o", ms=3.2, zorder=4)

    base, head = ys[0], ys[-1]
    ax.axhline(base, color=COLORS["base"], ls=":", lw=1.1, zorder=2)
    ax.axhline(head, color="#2f7d5c", ls=":", lw=1.1, zorder=2)
    ax.text(0, base + 190, f"{base:.0f} ms", ha="center", fontsize=9,
            fontweight="bold", color="#3a3f44")
    ax.text(24, head + 190, f"{head:.0f} ms", ha="center", fontsize=9,
            fontweight="bold", color="#2f7d5c")

    # Bracket the whole drop on the right, outside the bars.
    ax.annotate("", xy=(25.6, base), xytext=(25.6, head),
                arrowprops=dict(arrowstyle="<->", color="#3a3f44", lw=1.1))
    ax.text(25.9, (base + head) / 2, "−50.3%\n2.01x", va="center", ha="left",
            fontsize=10, fontweight="bold", color="#2f7d5c")

    # Nudge the first two clear of the taller bar on their left.
    for n, note, dx in ((2, "A2 −12.6%", 30), (5, "A5 −18.4%", 30),
                        (11, "A11 −7.9%", 0)):
        ax.annotate(note, (n, L.median(n)), textcoords="offset points",
                    xytext=(dx, 20), ha="center", fontsize=9, fontweight="bold",
                    color="#1d5540",
                    arrowprops=dict(arrowstyle="-", color="#1d5540", lw=0.8,
                                    shrinkA=1, shrinkB=3))
    ax.annotate("A6 +2.9%\n唯一变慢", (6, L.median(6)), textcoords="offset points",
                xytext=(58, 46), ha="center", fontsize=8.5, color=COLORS["slower"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=COLORS["slower"], lw=0.9))

    ax.set_xticks(xs)
    ax.set_xticklabels([f"S{n}" for n in xs], rotation=0, fontsize=7.6)
    ax.set_xlim(-0.9, 27.2)
    ax.set_ylim(0, base * 1.22)
    ax.set_ylabel("单步耗时中位数 (ms)")
    ax.set_title("MXFP4 累积消融阶梯：step time 从 12266 ms 降到 6097 ms")
    ax.set_xlabel("Sn = S(n−1) + 打开第 n 项优化（按引入时间排序）")
    ax.legend(handles=[Patch(facecolor=COLORS[k], label=v) for k, v in LEGEND],
              loc="upper right", ncols=5, bbox_to_anchor=(1.0, -0.10))
    fig.savefig(out / "ladder_steptime.png")
    plt.close(fig)


def fig_contributions(L: Ladder, out: pathlib.Path) -> None:
    rungs = sorted(L.arms[1:], key=L.step_pct)
    vals = [L.step_pct(n) for n in rungs]
    cats = [L.category(n) for n in rungs]

    fig, ax = plt.subplots(figsize=(11, 8))
    y = range(len(rungs))
    ax.barh(list(y), vals, color=[COLORS[c] for c in cats], height=0.72, zorder=3)
    ax.axvline(0, color="#2b2f33", lw=1.0, zorder=4)
    ax.axvspan(-NOISE_PCT, NOISE_PCT, color="#f2f3f4", zorder=1)
    ax.text(6.3, -0.85, "灰带 = ±1% 噪声，单次运行分不清真伪", fontsize=8.5,
            color="#84898e", va="center", ha="right")

    for i, v in enumerate(vals):
        off = -0.35 if v < 0 else 0.35
        ax.text(v + off, i, f"{v:+.1f}%".replace("-", "−"), va="center",
                ha="right" if v < 0 else "left", fontsize=8.6)

    ax.set_yticks(list(y))
    ax.set_yticklabels([L.label[n] for n in rungs], fontsize=8.4)
    ax.set_xlim(-21, 6.5)
    ax.set_ylim(len(rungs) - 0.4, -1.4)
    ax.set_xlabel("该级单步变化（负 = 变快）")
    ax.set_title("每一级各贡献了多少（按幅度排序）")
    ax.legend(handles=[Patch(facecolor=COLORS[k], label=v) for k, v in LEGEND],
              loc="lower left", ncols=2, bbox_to_anchor=(0.02, 0.03))
    fig.savefig(out / "ladder_contributions.png")
    plt.close(fig)


def fig_waterfall(L: Ladder, out: pathlib.Path) -> None:
    """Group the rungs and add up the actual ms each group moved."""
    deltas = {n: L.median(n) - L.median(n - 1) for n in L.arms[1:]}
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    w = 0.62
    labels = ["S0\n起点"]
    cursor = L.median(0)
    ax.bar(0, cursor, color=COLORS["base"], width=w, zorder=3)
    ax.text(0, cursor + 150, f"{cursor:.0f}", ha="center", fontsize=9,
            fontweight="bold", color="#3a3f44")

    for i, (name, arms, color) in enumerate(GROUPS, start=1):
        d = sum(deltas[n] for n in arms)
        bottom = min(cursor, cursor + d)
        ax.bar(i, max(abs(d), 28), bottom=bottom, color=color, width=w, zorder=3)
        ax.plot([i - 1 + w / 2, i - w / 2], [cursor, cursor], color="#9aa0a5",
                lw=0.9, ls="--", zorder=2)
        pct = d / L.median(0) * 100
        ax.text(i, max(cursor, cursor + d) + 150,
                f"{d:+.0f} ms\n({pct:+.1f} pt)", ha="center", fontsize=8.8,
                color=color if abs(pct) > 1 else "#84898e", fontweight="bold")
        cursor += d
        labels.append(name)

    last = len(GROUPS) + 1
    ax.plot([last - 1 + w / 2, last - w / 2], [cursor, cursor], color="#9aa0a5",
            lw=0.9, ls="--", zorder=2)
    ax.bar(last, cursor, color="#2f7d5c", width=w, zorder=3)
    ax.text(last, cursor + 150, f"{cursor:.0f}", ha="center",
            fontsize=9, fontweight="bold", color="#2f7d5c")
    labels.append("S24\n= HEAD")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("单步耗时中位数 (ms)")
    ax.set_ylim(0, L.median(0) * 1.13)
    ax.set_title("时间是从哪里省下来的：按类别拆 6170 ms")
    ax.text(0.5, -0.19, "括号内是占 S0 的百分点；各级 ms 差可加，报告正文的百分比是连乘口径。",
            transform=ax.transAxes, ha="center", fontsize=8.4, color="#84898e")
    fig.savefig(out / "group_waterfall.png")
    plt.close(fig)


def fig_asm_unlock(L: Ladder, out: pathlib.Path) -> None:
    arms = [n for n in L.arms if L.backends(n)]
    order = ["asm", "shuffled", "plain"]
    palette = {"asm": "#2f7d5c", "shuffled": "#dd8b2a", "plain": "#b9bcc0"}
    pretty = {"asm": "ASM kernel", "shuffled": "preshuffle", "plain": "Triton / plain"}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.12))
    bottom = [0.0] * len(arms)
    for key in order:
        vals = [L.backends(n).get(key, 0) for n in arms]
        ax.bar(arms, vals, bottom=bottom, color=palette[key], width=0.72,
               label=pretty[key], zorder=3)
        bottom = [b + v for b, v in zip(bottom, vals)]

    first = arms[0]
    ax.axvspan(-0.9, first - 0.5, color="#f4f5f6", zorder=1)
    ax.text((first - 1.4) / 2, 5.5, "A8 之前没有 autotune cache\n由静态字节阈值决定",
            ha="center", va="center", fontsize=8.6, color="#84898e")

    for n, note in ((8, "A8 实测 autotune\n1/11"), (12, "A12 tuned 表\n8/11"),
                    (17, "A17 swizzle 缓存\n11/11")):
        ax.annotate(note, (n, 11.6), ha="center", va="bottom", fontsize=8.4,
                    fontweight="bold", color="#1d5540")
    ax.set_ylim(0, 15.4)
    ax.set_yticks([0, 4, 8, 11])
    ax.set_ylabel("11 个 GEMM shape 的选择")
    ax.set_title("ASM backend 是被 autotune、tuned 表和 swizzle 缓存逐步解锁的")
    ax.legend(loc="upper left", ncols=1)

    ax2.plot(L.arms, [L.median(n) for n in L.arms], color="#2b2f33", lw=1.5,
             marker="o", ms=3.2, zorder=3)
    for n in (8, 12, 17):
        ax2.scatter([n], [L.median(n)], s=58, color="#2f7d5c", zorder=4)
    ax2.set_ylabel("单步耗时 (ms)")
    ax2.set_xlabel("A7 单独进 dispatch 只拿到 1/11——它需要另外三项才生效（报告 4.2）")
    ax2.set_xticks(L.arms)
    ax2.set_xticklabels([f"S{n}" for n in L.arms], fontsize=7.6)
    ax2.set_xlim(-0.8, 24.8)
    fig.savefig(out / "asm_unlock.png")
    plt.close(fig)


def fig_s18(L: Ladder, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    top = 0.0
    for n, color, lw in ((17, "#b9bcc0", 1.3), (24, "#4a72b0", 1.3),
                         (18, "#dd8b2a", 1.9)):
        pts = L.iters(n)
        ax.plot([i for i, _ in pts], [v for _, v in pts], color=color, lw=lw,
                marker="o", ms=2.6, label=f"S{n}", zorder=3 if n != 18 else 4)
        med = statistics.median(v for _, v in pts)
        ax.axhline(med, color=color, ls=":", lw=1.0, zorder=2)
        top = max(top, max(v for _, v in pts))

    s18 = L.summary[18]
    ax.set_ylim(top=top * 1.14)
    ax.annotate(f"S18 中位数 {s18['median_ms']:.0f} ms（只看好步）\n"
                f"S18 均值 {s18['mean_ms']:.0f} ms（含停顿）",
                xy=(0.985, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=9.5, color="#a3651c", fontweight="bold")
    ax.set_xlabel("iteration（已丢弃前 15 步的编译与探测）")
    ax.set_ylabel("单步耗时 (ms)")
    ax.set_title("S18 的 13/45 步停顿：中位数说 −0.7%，均值说 +9%")
    ax.legend(loc="upper left", ncols=3)
    fig.savefig(out / "s18_stalls.png")
    plt.close(fig)


def fig_loss(L: Ladder, out: pathlib.Path) -> None:
    xs = L.arms
    ys = [L.summary[n]["final_loss"] for n in xs]
    lo, hi, mid = min(ys), max(ys), statistics.median(ys)

    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.axhspan(lo, hi, color="#eaf1ed", zorder=1)
    ax.axhline(mid, color="#2f7d5c", ls=":", lw=1.1, zorder=2)
    ax.plot(xs, ys, color="#2f7d5c", lw=1.4, marker="o", ms=4, zorder=3)
    for n in (14, 18, 19):
        ax.scatter([n], [L.summary[n]["final_loss"]], s=70, facecolor="white",
                   edgecolor="#dd8b2a", lw=1.7, zorder=4)
    ax.annotate(f"注意 y 轴总跨度只有 {hi - lo:.4f} nats——圈出的三项是 Phase 0 预告会动 loss 的"
                "（A14 移 RNG 流、A18 少一轮量化、A19 SR 重抽签）",
                xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
                fontsize=8.4, color="#84898e")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"S{n}" for n in xs], fontsize=7.6)
    ax.set_xlim(-0.8, 24.8)
    ax.set_ylabel("第 60 步 loss")
    ax.set_title(f"loss 中性：25 个 arm 落在 {lo:.4f}–{hi:.4f}，全距 {(hi/lo-1)*100:.2f}%")
    fig.savefig(out / "loss_neutrality.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=pathlib.Path, default=PLOT_ROOT)
    ap.add_argument("--out", type=pathlib.Path, default=DOC_CHARTS)
    args = ap.parse_args()

    style()
    L = Ladder(args.ladder)
    if not L.arms:
        raise SystemExit(f"no arm summaries under {args.ladder}")
    args.out.mkdir(parents=True, exist_ok=True)

    fig_staircase(L, args.out)
    fig_contributions(L, args.out)
    fig_waterfall(L, args.out)
    fig_asm_unlock(L, args.out)
    fig_s18(L, args.out)
    fig_loss(L, args.out)
    print(f"wrote {len(list(args.out.glob('*.png')))} figures to {args.out}")


if __name__ == "__main__":
    main()
