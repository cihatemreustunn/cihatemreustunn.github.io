"""Generate the figures for the "Gaussian Processes from Scratch" blog post.

Everything here is plain NumPy - the same code that appears in the post.
Figures are written as SVG with the text left as *text* (svg.fonttype = 'none')
and the font-family rewritten to the site's stack, so the labels in the plots
are rendered by the browser in exactly the same font as the surrounding prose.

    python3 make_figures.py

Requires: numpy, matplotlib (scikit-learn only for the library-comparison figure).
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Styling: dark, transparent background, site font, site accent colour.
# ----------------------------------------------------------------------------
FONT_STACK = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
              "'Helvetica Neue',Arial,sans-serif")

FG = "#cccccc"       # body text colour of the site
MUTED = "#8a8a8a"
GRID = "#454545"
BLUE = "#4a9eff"     # site accent
AMBER = "#ffb454"
GREEN = "#5fd39a"
PINK = "#ff7eb6"
PURPLE = "#b39ddb"
SAMPLE_COLORS = [BLUE, AMBER, GREEN, PINK, PURPLE]

matplotlib.rcParams.update({
    "svg.fonttype": "none",
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "text.color": FG,
    "axes.labelcolor": FG,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.size": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "lines.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    """Save a figure as a responsive, font-inheriting SVG."""
    path = os.path.join(OUT, name)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    with open(path, "r", encoding="utf-8") as fh:
        svg = fh.read()

    # 1. use the website's font stack for every text element
    svg = re.sub(r"font-family: [^;]+;", "font-family: %s;" % FONT_STACK, svg)
    # 2. drop the fixed pt size so the SVG scales with its container
    svg = re.sub(r'(<svg[^>]*?)width="[\d.]+pt" height="[\d.]+pt"', r"\1", svg, count=1)
    svg = svg.replace("<svg ", '<svg preserveAspectRatio="xMidYMid meet" ', 1)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print("wrote", os.path.relpath(path, HERE))


def style_legend(ax, **kw):
    leg = ax.legend(facecolor="#363636", edgecolor=GRID, framealpha=0.85,
                    labelcolor=FG, **kw)
    return leg


# ----------------------------------------------------------------------------
# The Gaussian process machinery (pure NumPy).
# ----------------------------------------------------------------------------
def rbf(X1, X2, lengthscale=1.0, variance=1.0):
    """Squared-exponential (RBF) covariance between two sets of inputs."""
    X1 = np.atleast_2d(X1).reshape(-1, 1)
    X2 = np.atleast_2d(X2).reshape(-1, 1)
    sq = (X1 - X2.T) ** 2
    return variance * np.exp(-0.5 * sq / lengthscale ** 2)


def matern32(X1, X2, lengthscale=1.0, variance=1.0):
    X1 = np.atleast_2d(X1).reshape(-1, 1)
    X2 = np.atleast_2d(X2).reshape(-1, 1)
    d = np.abs(X1 - X2.T) / lengthscale
    return variance * (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)


def matern12(X1, X2, lengthscale=1.0, variance=1.0):
    X1 = np.atleast_2d(X1).reshape(-1, 1)
    X2 = np.atleast_2d(X2).reshape(-1, 1)
    d = np.abs(X1 - X2.T) / lengthscale
    return variance * np.exp(-d)


def periodic(X1, X2, lengthscale=1.0, period=2.0, variance=1.0):
    X1 = np.atleast_2d(X1).reshape(-1, 1)
    X2 = np.atleast_2d(X2).reshape(-1, 1)
    d = np.abs(X1 - X2.T)
    return variance * np.exp(-2 * np.sin(np.pi * d / period) ** 2 / lengthscale ** 2)


def linear(X1, X2, variance=1.0, offset=0.0):
    X1 = np.atleast_2d(X1).reshape(-1, 1)
    X2 = np.atleast_2d(X2).reshape(-1, 1)
    return variance * (X1 - offset) * (X2 - offset).T


def rational_quadratic(X1, X2, lengthscale=1.0, alpha=1.0, variance=1.0):
    X1 = np.atleast_2d(X1).reshape(-1, 1)
    X2 = np.atleast_2d(X2).reshape(-1, 1)
    sq = (X1 - X2.T) ** 2
    return variance * (1 + sq / (2 * alpha * lengthscale ** 2)) ** (-alpha)


def sample_gp(mean, cov, n_samples=1, rng=None, jitter=1e-9):
    """Draw samples from N(mean, cov) via the Cholesky factor."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = cov.shape[0]
    L = np.linalg.cholesky(cov + jitter * np.eye(n))
    u = rng.standard_normal((n, n_samples))
    return mean.reshape(-1, 1) + L @ u


def gp_posterior(X_s, X, y, kernel, noise=1e-8, **kernel_kwargs):
    """Posterior mean and covariance of a zero-mean GP at the test inputs X_s."""
    K = kernel(X, X, **kernel_kwargs) + noise ** 2 * np.eye(len(X))
    K_s = kernel(X, X_s, **kernel_kwargs)
    K_ss = kernel(X_s, X_s, **kernel_kwargs)

    L = np.linalg.cholesky(K + 1e-10 * np.eye(len(X)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    v = np.linalg.solve(L, K_s)

    mu = K_s.T @ alpha
    cov = K_ss - v.T @ v
    return mu.ravel(), cov


def nll(X, y, kernel, noise, **kernel_kwargs):
    """Negative log marginal likelihood."""
    n = len(X)
    K = kernel(X, X, **kernel_kwargs) + noise ** 2 * np.eye(n)
    L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return float(0.5 * y.ravel() @ alpha.ravel()
                 + np.sum(np.log(np.diag(L)))
                 + 0.5 * n * np.log(2 * np.pi))


def plot_gp(ax, X_s, mu, cov, X=None, y=None, samples=0, rng=None,
            band_color=BLUE, mean_label="posterior mean", band_label="±2σ"):
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    ax.fill_between(X_s.ravel(), mu - 2 * std, mu + 2 * std,
                    color=band_color, alpha=0.18, linewidth=0, label=band_label)
    ax.plot(X_s, mu, color=band_color, label=mean_label)
    if samples:
        S = sample_gp(mu, cov, samples, rng=rng)
        for i in range(samples):
            ax.plot(X_s, S[:, i], lw=1.0, alpha=0.75,
                    color=SAMPLE_COLORS[(i + 1) % len(SAMPLE_COLORS)])
    if X is not None:
        ax.plot(X, y, "o", color="#ffffff", markersize=5.5,
                markeredgecolor="#2b2b2b", markeredgewidth=1.0, zorder=5,
                label="observations")
    return std


# ============================================================================
# Figure 1 - marginalisation and conditioning of a 2D Gaussian
# ============================================================================
def fig_marginal_conditional():
    rng = np.random.default_rng(3)
    mu = np.array([0.0, 0.0])
    rho = 0.85
    Sigma = np.array([[1.0, rho], [rho, 1.0]])

    g = np.linspace(-3.2, 3.2, 140)
    A, B = np.meshgrid(g, g)
    pos = np.stack([A, B], axis=-1)
    Si = np.linalg.inv(Sigma)
    q = np.einsum("...i,ij,...j->...", pos - mu, Si, pos - mu)
    dens = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    from matplotlib.colors import LinearSegmentedColormap
    fade = LinearSegmentedColormap.from_list(
        "fade", [(0.29, 0.62, 1.0, 0.0), (0.29, 0.62, 1.0, 0.85)])

    ax = axes[0]
    ax.contourf(A, B, dens, levels=12, cmap=fade)
    ax.contour(A, B, dens, levels=6, colors="#ffffff", linewidths=0.5, alpha=0.35)
    pts = rng.multivariate_normal(mu, Sigma, 120)
    ax.plot(pts[:, 0], pts[:, 1], ".", color="#ffffff", alpha=0.45, markersize=3)
    ax.set_xlabel("f(x₁)")
    ax.set_ylabel("f(x₂)")
    ax.set_title("joint  p(f₁, f₂),  correlation 0.85")

    ax = axes[1]
    ax.plot(g, np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi), color=BLUE,
            label="p(f₂)  —  marginal")
    ax.fill_between(g, 0, np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi),
                    color=BLUE, alpha=0.15, linewidth=0)
    ax.set_xlabel("f(x₂)")
    ax.set_ylabel("density")
    ax.set_title("marginalising:  integrate f₁ away")
    style_legend(ax, loc="upper left")

    ax = axes[2]
    obs = 1.5
    m_c = rho * obs
    s_c = np.sqrt(1 - rho ** 2)
    ax.plot(g, np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi), color=MUTED,
            ls="--", label="p(f₂)  before")
    cond = np.exp(-0.5 * ((g - m_c) / s_c) ** 2) / (s_c * np.sqrt(2 * np.pi))
    ax.plot(g, cond, color=AMBER, label="p(f₂ | f₁ = 1.5)")
    ax.fill_between(g, 0, cond, color=AMBER, alpha=0.18, linewidth=0)
    ax.axvline(m_c, color=AMBER, lw=0.8, alpha=0.6)
    ax.set_xlabel("f(x₂)")
    ax.set_title("conditioning:  observe f₁ = 1.5")
    style_legend(ax, loc="upper left")

    fig.tight_layout()
    save(fig, "fig1-marginal-conditional.svg")


# ============================================================================
# Figure 2 - two function values, near and far apart
# ============================================================================
def fig_two_points():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    X_s = np.linspace(-3, 3, 160)
    K = rbf(X_s, X_s, lengthscale=1.0)
    S = sample_gp(np.zeros(len(X_s)), K, 5, rng=rng)
    x1, x2, x3 = -1.0, -0.6, 2.0
    for i in range(5):
        axes[0].plot(X_s, S[:, i], lw=1.2, alpha=0.9, color=SAMPLE_COLORS[i])
    for c, x in zip([AMBER, AMBER, PINK], [x1, x2, x3]):
        axes[0].axvline(x, color=c, lw=0.9, ls="--", alpha=0.8)
    axes[0].set_title("five random functions, ℓ = 1")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_ylim(-3.2, 3.2)
    axes[0].text(-0.8, 2.7, "x₁, x₂", color=AMBER, ha="center", fontsize=9)
    axes[0].text(2.0, 2.7, "x₃", color=PINK, ha="center", fontsize=9)

    for ax, (xa, xb, lab, col) in zip(axes[1:], [
            (x1, x2, "close together:  |x₁ − x₂| = 0.4", AMBER),
            (x1, x3, "far apart:  |x₁ − x₃| = 3.0", PINK)]):
        Kp = rbf(np.array([xa, xb]), np.array([xa, xb]), lengthscale=1.0)
        pts = rng.multivariate_normal([0, 0], Kp, 400)
        ax.plot(pts[:, 0], pts[:, 1], ".", color=col, alpha=0.45, markersize=3.5)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal")
        ax.set_xlabel("f(x₁)")
        ax.set_ylabel("f(x') ")
        ax.set_title("%s\nk = %.2f" % (lab, Kp[0, 1]))

    fig.tight_layout()
    save(fig, "fig2-two-points.svg")


# ============================================================================
# Figure 3 - the prior: lengthscale and amplitude
# ============================================================================
def fig_prior():
    rng = np.random.default_rng(11)
    X_s = np.linspace(-5, 5, 300)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
    for ax, l in zip(axes, [0.3, 1.0, 3.0]):
        K = rbf(X_s, X_s, lengthscale=l)
        std = np.sqrt(np.diag(K))
        ax.fill_between(X_s, -2 * std, 2 * std, color=BLUE, alpha=0.13, linewidth=0)
        S = sample_gp(np.zeros(len(X_s)), K, 4, rng=np.random.default_rng(11))
        for i in range(4):
            ax.plot(X_s, S[:, i], lw=1.3, alpha=0.9,
                    color=[AMBER, GREEN, PINK, PURPLE][i])
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_title("ℓ = %.1f" % l)
        ax.set_xlabel("x")
        ax.set_ylim(-3.2, 3.2)
    axes[0].set_ylabel("f(x)")
    fig.suptitle("Samples from the prior  f ~ GP(0, k)   —   shaded band is ±2σ",
                 color=FG, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig3-prior-samples.svg")


# ============================================================================
# Figure 4 - kernel gallery: covariance matrix and the functions it produces
# ============================================================================
def fig_kernels():
    X_s = np.linspace(-4, 4, 240)
    specs = [
        ("RBF\nℓ = 1", lambda a, b: rbf(a, b, 1.0)),
        ("Matérn 3/2\nℓ = 1", lambda a, b: matern32(a, b, 1.0)),
        ("Exponential\n(Matérn 1/2)", lambda a, b: matern12(a, b, 1.0)),
        ("Periodic\np = 2", lambda a, b: periodic(a, b, 1.0, 2.0)),
        ("Rational quadratic\nα = 0.5", lambda a, b: rational_quadratic(a, b, 1.0, 0.5)),
        ("Linear", lambda a, b: linear(a, b, 0.35)),
    ]
    fig, axes = plt.subplots(2, 6, figsize=(13.5, 4.6),
                             gridspec_kw={"height_ratios": [1, 1.15]})
    for j, (name, kfun) in enumerate(specs):
        K = kfun(X_s, X_s)
        ax = axes[0, j]
        ax.imshow(K, cmap="magma", origin="lower", extent=[-4, 4, -4, 4])
        ax.set_title(name, fontsize=10)
        ax.grid(False)
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([-4, 0, 4])
        if j == 0:
            ax.set_ylabel("x'")

        ax = axes[1, j]
        S = sample_gp(np.zeros(len(X_s)), K, 3, rng=np.random.default_rng(4))
        for i in range(3):
            ax.plot(X_s, S[:, i], lw=1.2, color=SAMPLE_COLORS[i], alpha=0.9)
        ax.set_xlabel("x")
        ax.set_xticks([-4, 0, 4])
        if j == 0:
            ax.set_ylabel("f(x)")
    axes[0, 0].set_xlabel("x", labelpad=1)
    fig.suptitle("Top: the covariance matrix k(x, x′).    Bottom: three functions drawn from it.",
                 color=FG, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig4-kernel-gallery.svg")


# ============================================================================
# Figure 5 - the posterior tightens as data arrives (noise-free)
# ============================================================================
def truth(x):
    return np.sin(x) + 0.5 * np.sin(3 * x)


def fig_posterior_growth():
    X_s = np.linspace(-5, 5, 300)
    all_x = np.array([-4.2, 1.7, -1.0, 4.1, -2.6, 0.4, 2.9, -3.4])
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2), sharey=True)
    for ax, n in zip(axes, [0, 1, 3, 8]):
        rng = np.random.default_rng(2)
        if n == 0:
            K = rbf(X_s, X_s, lengthscale=1.0)
            plot_gp(ax, X_s, np.zeros(len(X_s)), K, samples=3, rng=rng,
                    mean_label="prior mean")
        else:
            X = np.sort(all_x[:n])
            y = truth(X)
            mu, cov = gp_posterior(X_s, X, y, rbf, noise=1e-6, lengthscale=1.0)
            plot_gp(ax, X_s, mu, cov, X, y, samples=3, rng=rng)
        ax.plot(X_s, truth(X_s), color="#ffffff", ls="--", lw=1.1, alpha=0.55,
                label="true f")
        ax.set_title("%d observation%s" % (n, "" if n == 1 else "s"))
        ax.set_xlabel("x")
        ax.set_ylim(-3.0, 3.0)
    axes[0].set_ylabel("f(x)")
    style_legend(axes[-1], loc="lower right")
    fig.tight_layout()
    save(fig, "fig5-posterior-growth.svg")


# ============================================================================
# Figure 6 - observation noise
# ============================================================================
def fig_noise():
    rng = np.random.default_rng(5)
    X = np.sort(rng.uniform(-4.5, 4.5, 14))
    y_clean = truth(X)
    y = y_clean + 0.25 * rng.standard_normal(len(X))
    X_s = np.linspace(-5.5, 5.5, 300)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), sharey=True)
    for ax, s in zip(axes, [1e-4, 0.25, 1.0]):
        mu, cov = gp_posterior(X_s, X, y, rbf, noise=s, lengthscale=1.0)
        plot_gp(ax, X_s, mu, cov, X, y, samples=0, rng=rng)
        ax.plot(X_s, truth(X_s), color="#ffffff", ls="--", lw=1.1, alpha=0.5,
                label="true f")
        ax.set_title("σₙ = %s" % ("0 (interpolation)" if s < 1e-3 else "%.2f" % s))
        ax.set_xlabel("x")
        ax.set_ylim(-3.2, 3.2)
    axes[0].set_ylabel("y")
    style_legend(axes[-1], loc="lower right")
    fig.suptitle("Same data, three assumptions about how noisy it is  (true σₙ = 0.25)",
                 color=FG, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig6-noise.svg")


# ============================================================================
# Figure 7 - hyperparameters and the marginal likelihood
# ============================================================================
def fig_marginal_likelihood():
    rng = np.random.default_rng(5)
    X = np.sort(rng.uniform(-4.5, 4.5, 14))
    y = truth(X) + 0.25 * rng.standard_normal(len(X))
    X_s = np.linspace(-5.5, 5.5, 300)

    ls = np.logspace(-1.2, 1.0, 70)
    ns = np.logspace(-2.2, 0.3, 70)
    Z = np.empty((len(ns), len(ls)))
    for i, s in enumerate(ns):
        for j, l in enumerate(ls):
            Z[i, j] = nll(X, y, rbf, s, lengthscale=l)
    i_opt, j_opt = np.unravel_index(np.argmin(Z), Z.shape)
    l_opt, s_opt = ls[j_opt], ns[i_opt]

    fig = plt.figure(figsize=(12.5, 4.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    Zc = np.clip(Z, None, Z.min() + 45)
    cs = ax.contourf(ls, ns, Zc, levels=np.linspace(Z.min(), Z.min() + 45, 22),
                     cmap="magma_r", extend="max")
    ax.contour(ls, ns, Zc, levels=np.linspace(Z.min(), Z.min() + 45, 11),
               colors="#ffffff", linewidths=0.4, alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(l_opt, s_opt, "*", color=BLUE, markersize=16,
            markeredgecolor="#2b2b2b", label="minimum")
    for l, c in zip([0.15, l_opt, 5.0], [AMBER, BLUE, GREEN]):
        ax.plot(l, s_opt, "o", color=c, markersize=6, markeredgecolor="#2b2b2b")
    ax.set_xlabel("lengthscale ℓ")
    ax.set_ylabel("noise σₙ")
    ax.set_title("negative log marginal likelihood")
    ax.grid(False)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=MUTED)
    cb.outline.set_edgecolor(GRID)

    titles = ["too short: ℓ = 0.15\n(overfits, no structure)",
              "learned: ℓ = %.2f, σₙ = %.2f\n(NLL minimum)" % (l_opt, s_opt),
              "too long: ℓ = 5\n(underfits, too smooth)"]
    for k, (l, c, t) in enumerate(zip([0.15, l_opt, 5.0], [AMBER, BLUE, GREEN], titles)):
        ax = fig.add_subplot(gs[0, k + 1])
        mu, cov = gp_posterior(X_s, X, y, rbf, noise=s_opt, lengthscale=l)
        plot_gp(ax, X_s, mu, cov, X, y, band_color=c)
        ax.plot(X_s, truth(X_s), color="#ffffff", ls="--", lw=1.0, alpha=0.5)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylim(-3.4, 3.4)
        if k == 0:
            ax.set_ylabel("y")
    fig.tight_layout()
    save(fig, "fig7-marginal-likelihood.svg")
    return l_opt, s_opt


# ============================================================================
# Figure 8 - NumPy implementation vs scikit-learn
# ============================================================================
def fig_library_comparison():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    rng = np.random.default_rng(5)
    X = np.sort(rng.uniform(-4.5, 4.5, 14))
    y = truth(X) + 0.25 * rng.standard_normal(len(X))
    X_s = np.linspace(-5.5, 5.5, 300)

    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
              + WhiteKernel(0.1, (1e-6, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                   n_restarts_optimizer=8, random_state=0)
    gpr.fit(X.reshape(-1, 1), y)
    mu_sk, std_sk = gpr.predict(X_s.reshape(-1, 1), return_std=True)

    k = gpr.kernel_
    var = k.k1.k1.constant_value
    ell = k.k1.k2.length_scale
    noise = np.sqrt(k.k2.noise_level)

    mu_np, cov_np = gp_posterior(X_s, X, y, rbf, noise=noise,
                                 lengthscale=ell, variance=var)
    std_np = np.sqrt(np.clip(np.diag(cov_np), 0, None))

    std_y = np.sqrt(std_np ** 2 + noise ** 2)  # sklearn's std includes the noise

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.fill_between(X_s, mu_sk - 2 * std_sk, mu_sk + 2 * std_sk, color=AMBER,
                    alpha=0.18, linewidth=0, label="scikit-learn ±2σ")
    ax.plot(X_s, mu_sk, color=AMBER, lw=3.0, alpha=0.8, label="scikit-learn mean")
    ax.plot(X_s, mu_np, color=BLUE, lw=1.4, ls="--", label="NumPy mean")
    ax.plot(X_s, mu_np + 2 * std_y, color=BLUE, lw=0.9, ls="--", alpha=0.8,
            label="NumPy ±2σ")
    ax.plot(X_s, mu_np - 2 * std_y, color=BLUE, lw=0.9, ls="--", alpha=0.8)
    ax.plot(X, y, "o", color="#ffffff", markersize=5.5,
            markeredgecolor="#2b2b2b", zorder=5, label="observations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("same hyperparameters ⇒ same posterior\nℓ = %.3f,  σf = %.3f,  σₙ = %.3f"
                 % (ell, np.sqrt(var), noise), fontsize=10)
    style_legend(ax, loc="lower right", ncol=2)

    ax = axes[1]
    ax.semilogy(X_s, np.abs(std_sk - std_np) + 1e-18, color=PINK,
                label="|Δ std| vs std of f")
    ax.semilogy(X_s, np.abs(mu_sk - mu_np) + 1e-18, color=BLUE, label="|Δ mean|")
    ax.semilogy(X_s, np.abs(std_sk - std_y) + 1e-18, color=GREEN,
                label="|Δ std| vs std of y")
    ax.set_ylim(1e-18, 1e1)
    ax.set_xlabel("x")
    ax.set_ylabel("absolute difference")
    ax.set_title("identical — once you compare like with like", fontsize=10)
    style_legend(ax, loc="upper right")
    fig.tight_layout()
    save(fig, "fig8-library-comparison.svg")
    return dict(lengthscale=ell, sigma_f=np.sqrt(var), sigma_n=noise,
                d_mean=float(np.max(np.abs(mu_sk - mu_np))),
                d_std_latent=float(np.max(np.abs(std_sk - std_np))),
                d_std_y=float(np.max(np.abs(std_sk - std_y))),
                lml=float(gpr.log_marginal_likelihood_value_),
                my_nll=-nll(X, y, rbf, noise, lengthscale=ell, variance=var))


# ============================================================================
# Figure 9 - Bayesian optimisation with expected improvement
# ============================================================================
def fig_bayesopt():
    def objective(x):
        return -(np.sin(3 * x) + 0.4 * x ** 2 - 0.6 * x)

    def erf_vec(t):
        # Abramowitz & Stegun 7.1.26 - keeps this file NumPy-only
        s = np.sign(t)
        t = np.abs(t)
        a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
        p = 0.3275911
        k = 1.0 / (1.0 + p * t)
        yv = 1.0 - (((((a[4] * k + a[3]) * k) + a[2]) * k + a[1]) * k + a[0]) * k * np.exp(-t * t)
        return s * yv

    def expected_improvement(mu, std, best, xi=0.01):
        std = np.maximum(std, 1e-9)
        imp = mu - best - xi
        z = imp / std
        Phi = 0.5 * (1 + erf_vec(z / np.sqrt(2)))
        phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        return imp * Phi + std * phi

    X_s = np.linspace(-2, 2, 400)
    X = np.array([-1.6, 0.1])
    y = objective(X)
    n_iter = 4

    fig, axes = plt.subplots(2, n_iter, figsize=(13.5, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for it in range(n_iter):
        mu, cov = gp_posterior(X_s, X, y, rbf, noise=1e-3,
                               lengthscale=0.40, variance=1.5)
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        ei = expected_improvement(mu, std, y.max())
        x_next = X_s[np.argmax(ei)]

        ax = axes[0, it]
        ax.fill_between(X_s, mu - 2 * std, mu + 2 * std, color=BLUE, alpha=0.18,
                        linewidth=0)
        ax.plot(X_s, mu, color=BLUE, label="GP mean")
        ax.plot(X_s, objective(X_s), color="#ffffff", ls="--", lw=1.0, alpha=0.5,
                label="objective")
        ax.plot(X, y, "o", color="#ffffff", markersize=5.5,
                markeredgecolor="#2b2b2b", zorder=5, label="evaluated")
        ax.axvline(x_next, color=GREEN, lw=1.0, ls=":")
        ax.plot(X_s[np.argmax(objective(X_s))], objective(X_s).max(), "*",
                color=AMBER, markersize=13, markeredgecolor="#2b2b2b", zorder=6,
                label="true optimum")
        ax.set_title("%d evaluations  —  best so far %.2f" % (len(X), y.max()),
                     fontsize=10)
        if it == 0:
            ax.set_ylabel("f(x)")
            style_legend(ax, loc="lower center", ncol=2)

        ax = axes[1, it]
        ax.fill_between(X_s, 0, ei, color=GREEN, alpha=0.3, linewidth=0)
        ax.plot(X_s, ei, color=GREEN, lw=1.2)
        ax.axvline(x_next, color=GREEN, lw=1.0, ls=":")
        ax.set_xlabel("x")
        if it == 0:
            ax.set_ylabel("expected\nimprovement")

        X = np.append(X, x_next)
        y = np.append(y, objective(x_next))

    fig.suptitle("Bayesian optimisation: the GP proposes where to look next",
                 color=FG, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig9-bayesopt.svg")


# ============================================================================
# Figure 10 - a combustion-flavoured example with extrapolation
# ============================================================================
def fig_flame_speed():
    """Synthetic laminar burning velocity data for an ammonia/hydrogen blend.

    The numbers are made up - the point is the shape of the uncertainty, not
    the chemistry.
    """
    rng = np.random.default_rng(21)

    def s_L(phi):
        return 14.0 * np.exp(-((phi - 1.08) ** 2) / (2 * 0.17 ** 2)) + 1.0

    phi_train = np.array([0.70, 0.78, 0.85, 0.92, 1.00, 1.08, 1.15, 1.22, 1.30])
    y_train = s_L(phi_train) + 0.45 * rng.standard_normal(len(phi_train))
    phi_s = np.linspace(0.55, 1.75, 300)

    mean = y_train.mean()
    mu, cov = gp_posterior(phi_s, phi_train, y_train - mean, rbf,
                           noise=0.45, lengthscale=0.22, variance=30.0)
    mu = mu + mean
    std = np.sqrt(np.clip(np.diag(cov), 0, None))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.axvspan(1.30, 1.75, color="#ffffff", alpha=0.05, linewidth=0)
    ax.fill_between(phi_s, mu - 2 * std, mu + 2 * std, color=BLUE, alpha=0.18,
                    linewidth=0, label="GP ±2σ")
    ax.plot(phi_s, mu, color=BLUE, label="GP mean")
    ax.plot(phi_s, s_L(phi_s), color="#ffffff", ls="--", lw=1.1, alpha=0.55,
            label="ground truth")
    ax.errorbar(phi_train, y_train, yerr=0.9, fmt="o", color="#ffffff",
                ecolor=MUTED, elinewidth=1.0, capsize=3, markersize=5.5,
                markeredgecolor="#2b2b2b", zorder=5, label="measurements")
    ax.text(1.52, 15.0, "no data here", color=MUTED, ha="center", fontsize=9)
    ax.annotate("", xy=(1.52, 13.0), xytext=(1.52, 14.4),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.set_xlabel("equivalence ratio φ")
    ax.set_ylabel("laminar burning velocity  (cm/s)")
    ax.set_title("A GP knows what it does not know", fontsize=11)
    ax.set_ylim(-6, 22)
    style_legend(ax, loc="upper left")
    fig.tight_layout()
    save(fig, "fig10-flame-speed.svg")


if __name__ == "__main__":
    fig_marginal_conditional()
    fig_two_points()
    fig_prior()
    fig_kernels()
    fig_posterior_growth()
    fig_noise()
    l_opt, s_opt = fig_marginal_likelihood()
    print("NLL minimum at lengthscale = %.3f, noise = %.3f" % (l_opt, s_opt))
    print("sklearn comparison:", fig_library_comparison())
    fig_bayesopt()
    fig_flame_speed()
