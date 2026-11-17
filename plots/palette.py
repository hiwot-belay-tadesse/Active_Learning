"""One colour per participant, stable across every figure in the repo.

Lifted out of query_hist_by_round.py so the cohort provenance figure gives a participant the
same colour as the per-target one; the selection is deterministic, so importing it here cannot
change a colour that was already published.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402


def _lab(rgb):
    """sRGB (n, 3) in [0, 1] -> CIELAB, so colours are compared the way an eye compares them.

    Euclidean distance in RGB is a poor proxy: tab20's blue/light-blue pair is far apart in RGB
    and nearly indistinguishable on the page, which is exactly the failure this palette is
    meant to avoid.
    """
    c = np.asarray(rgb, float)[:, :3]
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def distinct_colors(n):
    """-> n colours chosen to be as far apart as possible, in a fixed order.

    Taking the first n entries of tab20 + tab20b + tab20c (the previous choice) hands out
    tab20's light/dark PAIRS and tab20b/c's near-identical quartets, so neighbouring stack
    segments -- and, worse, the target's own block against its donors -- came out in shades of
    one hue. Instead, pool nine qualitative maps and pick greedily by farthest-point in
    CIELAB: each new colour is the candidate whose nearest already-chosen colour is furthest
    away. Deterministic, so a participant keeps their colour across figures and runs.

    Near-white and near-black candidates are dropped first: white collides with the segment
    edges and black with the self block's outline.
    """
    pool = np.vstack([plt.get_cmap(c)(np.linspace(0, 1, k)) for c, k in
                      (("tab20", 20), ("tab20b", 20), ("tab20c", 20), ("Set1", 9),
                       ("Set2", 8), ("Set3", 12), ("Dark2", 8), ("Accent", 8),
                       ("Paired", 12))])[:, :3]
    pool = np.unique(np.round(pool, 6), axis=0)
    L = _lab(pool)
    pool, L = pool[(L[:, 0] > 22) & (L[:, 0] < 93)], L[(L[:, 0] > 22) & (L[:, 0] < 93)]
    if n > len(pool):
        raise SystemExit(f"only {len(pool)} usable colours for {n} participants")
    start = int(np.argmax(np.linalg.norm(L - L.mean(axis=0), axis=1)))
    picked = [start]
    d = np.linalg.norm(L - L[start], axis=1)
    for _ in range(n - 1):
        j = int(np.argmax(d))
        picked.append(j)
        d = np.minimum(d, np.linalg.norm(L - L[j], axis=1))
    sep = min(np.min(np.linalg.norm(L[picked] - L[picked][i], axis=1)[
        np.arange(len(picked)) != i]) for i in range(len(picked)))
    print(f"palette: {n} colours from {len(pool)} candidates, "
          f"smallest pairwise CIELAB distance {sep:.1f}")
    return np.hstack([pool[picked], np.ones((n, 1))])


# A hand-checked high-contrast set (Trubetskoy's 20): every pair differs in hue, not only in
# lightness, so neighbouring bars never read as two shades of one colour. Use it when the number
# of categories is small enough to keep that promise -- distinct_colors() above degrades to
# shades past ~20 because no 43 colours are mutually distinguishable.
DISTINCT = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
            "#bfef45", "#469990", "#9a6324", "#000075", "#808000", "#800000", "#000000",
            "#fabed4", "#ffe119", "#aaffc3", "#dcbeff", "#a9a9a9", "#ffd8b1"]


def fixed_colors(n):
    """-> n mutually distinguishable colours, or a clear error if n is too large for that."""
    if n > len(DISTINCT):
        raise SystemExit(f"{n} categories: no {n} colours are mutually distinguishable; "
                         f"ask for at most {len(DISTINCT)} (--top) or accept shades")
    return DISTINCT[:n]


def bright_colors(n, min_chroma=45.0, verbose=True):
    """-> n saturated colours, mutually far apart AND far from grey.

    Two separate requirements, and the earlier golden-angle version only met the second. Stepping
    hue by 137.508 degrees spreads colours well ON AVERAGE but says nothing about the closest
    PAIR: at n=23 it put index 7 at hue 243 and index 20 at hue 230, which is M11 and M36 in the
    same blue. So: build a dense grid of candidates, throw away everything whose CIELAB chroma is
    below `min_chroma` (that is what keeps them clear of the grey of a de-emphasised bar), then
    pick greedily by farthest point in CIELAB, which maximises the minimum pairwise distance.
    """
    import colorsys
    cand = np.array([colorsys.hsv_to_rgb(h / 360.0, sv, vv)
                     for h in range(0, 360, 2)
                     for sv in (0.75, 0.9, 1.0)
                     for vv in (0.55, 0.75, 0.95)])
    L = _lab(cand)
    keep = (np.hypot(L[:, 1], L[:, 2]) >= min_chroma) & (L[:, 0] > 25) & (L[:, 0] < 92)
    cand, L = cand[keep], L[keep]
    if n > len(cand):
        raise SystemExit(f"only {len(cand)} saturated candidates for {n} categories")
    start = int(np.argmax(L[:, 0] * 0 + np.hypot(L[:, 1], L[:, 2])))   # the most chromatic one
    picked = [start]
    d = np.linalg.norm(L - L[start], axis=1)
    for _ in range(n - 1):
        j = int(np.argmax(d))
        picked.append(j)
        d = np.minimum(d, np.linalg.norm(L - L[j], axis=1))
    P = L[picked]
    sep = min(np.min(np.linalg.norm(P - P[i], axis=1)[np.arange(len(P)) != i])
              for i in range(len(P)))
    if verbose:
        print(f"highlight palette: {n} saturated colours, smallest pairwise CIELAB distance "
              f"{sep:.1f}, smallest chroma {np.hypot(P[:, 1], P[:, 2]).min():.0f}")
    return [tuple(c) for c in cand[picked]]
