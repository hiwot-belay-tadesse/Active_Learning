"""One colour per participant and per acquisition, stable across every figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402


def _lab(rgb):
    """sRGB (n, 3) in [0, 1] -> CIELAB."""
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
    """-> n colours, greedy farthest-point in CIELAB over nine qualitative maps."""
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


# Trubetskoy's 20: hand-checked, every pair differs in hue, not only in lightness.
DISTINCT = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
            "#bfef45", "#469990", "#9a6324", "#000075", "#808000", "#800000", "#000000",
            "#fabed4", "#ffe119", "#aaffc3", "#dcbeff", "#a9a9a9", "#ffd8b1"]


def fixed_colors(n):
    """-> n mutually distinguishable colours, or a clear error if n is too large for that."""
    if n > len(DISTINCT):
        raise SystemExit(f"{n} categories: no {n} colours are mutually distinguishable; "
                         f"ask for at most {len(DISTINCT)} (--top) or accept shades")
    return DISTINCT[:n]


# Shared by the grid, the curves and the provenance panels.
# EU is saddlebrown in every pool: `laplace` and `laplace_pool` are one acquisition, two pools.
METHOD_COLORS = {"laplace": "saddlebrown", "laplace_pool": "saddlebrown",
                 "laplace_people": "slategray", "laplace_var": "gold",
                 "coreset": "seagreen", "typiclust": "purple", "probcover": "dodgerblue",
                 "random": "darkorange", "uncertainty": "steelblue", "eer": "magenta"}

# Hues with different NAMES: no distance rule rules out a navy beside a mid blue.
BRIGHT = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
          "#f032e6", "#bfef45", "#469990", "#9a6324", "#808000", "#ffe119"]
MAX_BRIGHT = len(BRIGHT)


def bright_colors(n, verbose=True):
    """-> n saturated, mutually unmistakable colours; errors rather than resorting to shades."""
    if n > MAX_BRIGHT:
        raise SystemExit(f"{n} categories: only {MAX_BRIGHT} colours are distinct without "
                         f"repeating a hue at another lightness")
    if verbose:
        print(f"highlight palette: {n} of {MAX_BRIGHT} named hues")
    return BRIGHT[:n]
