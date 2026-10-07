import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy.interpolate import PchipInterpolator

def _apply_style(fontsize):
    """LaTeX serif text, one size for everything: 14 for Figs. 3-4, 20 for Figs. 5-6."""
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    plt.rcParams.update({key: fontsize for key in
                         ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})


# Figs. 5-6 are drawn at twice the 246 pt column width (hence 20 = 2 x 10 pt text);
# the Fig. 5 panels share the golden-ratio height of that width
FIG5_HEIGHT = (2*246/72.27) * ((5**0.5 - 1)/2)

# ...and are shown in the notebook at 70 dpi instead of 100, so their text appears at the on-screen
# size of Figs. 3-4 (20 pt at 70 dpi = 14 pt at 100 dpi). Only the display changes: layout is in
# points, and PDF output does not depend on dpi.
DPI_2X = 100 * 14/20


def smooth(y, n=500):
    """Monotone (PCHIP) interpolation through the points (i, y[i]); no fit, passes through every point.

    Params:
        - y (array): values on the integer grid 0, 1, ..., len(y)-1.
        - n (int): number of points returned.
    Returns:
        - xs, ys (ndarray, shape (n,)): the interpolated curve, ready for ax.plot(*smooth(y)).
    """
    x = np.arange(len(y))
    xs = np.linspace(x[0], x[-1], n)
    return xs, PchipInterpolator(x, y)(xs)


def _frame_legend(leg):
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.95)


def _frame_axes(*axes):
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")


# ===========================================
# ============== Figure 3 ===================
# ===========================================

def setup_fig3_format():
    _apply_style(14)
    fig = plt.figure(figsize=(8*0.8, 4*0.8))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.05, 0.95], height_ratios=[1.1, 0.9], wspace=0.33, hspace=0.08)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1], sharex=ax_b)

    # (a) one smooth line through the points, the points drawn on top
    styles = {}
    for key, color, marker in [('coherent', 'navy', 'o'), ('glueball', 'darkviolet', 's')]:
        styles[key] = {'line':    dict(linewidth=1.7, color=color, zorder=2),
                       'markers': dict(linestyle='none', zorder=3, color=color, marker=marker, markersize=4)}
    styles['mean'] = dict(linewidth=1.5, color='#A9A9A9', linestyle=(0, (1, 2)), capstyle='round', zorder=1)

    # (b, c) one style per method
    linestyles = ["solid", (0, (3, 3.5)), (0, (1, 2))]
    colors = ["tomato", "deepskyblue", "darkviolet"]
    method_labels = ["Exact", "Effective", "Analytic"]
    styles['methods'] = [dict(linewidth=2, color=c, linestyle=ls, dash_capstyle='round', label=lab)
                         for c, ls, lab in zip(colors, linestyles, method_labels)]

    return fig, (ax_a, ax_b, ax_c), styles


def finalize_fig3_format(fig, axes, styles):
    ax_a, ax_b, ax_c = axes
    ax = ax_a

    ax.set_ylim(-0.008, 0.195)

    ax.set_xlabel(r"$\mathcal{N}$")
    ax.set_ylabel(r"$P(\mathcal{N})$")

    handles = [Line2D([0], [0], linewidth=styles[key]['line']['linewidth'], color=styles[key]['markers']['color'],
                      marker=styles[key]['markers']['marker'], markersize=styles[key]['markers']['markersize'])
               for key in ('coherent', 'glueball')]
    leg = ax.legend(handles, ['Coherent\nstate', 'Glueball\ncondensate'],
                    frameon=True, handletextpad=0.45, borderpad=0.35,
                    handlelength=1.4, loc=[0.336,0.576])
    _frame_legend(leg)

    ax_b.set_ylabel("$\\langle \\mathcal{N}\\rangle$", labelpad=0)
    ax_c.set_ylabel("Im$\\langle B\\rangle$", labelpad=8)
    ax_c.set_xlabel("$\\varepsilon^2 t/U$")
    ax_b.set_xlim(0, 5)
    ax_b.set_ylim(0, 10.5)
    ax_b.set_xticks([0,1,2,3,4,5])
    ax_c.set_xlim(0, 5)
    ax_c.set_ylim(0, 9)
    ax_c.set_xticks([0,1,2,3,4,5])
    ax_b.tick_params(labelbottom=False)

    style_handles = [Line2D([0], [0], **s) for s in styles['methods']]
    leg = fig.legend(handles=style_handles, frameon=True, handletextpad=0.45, borderpad=0.35, handlelength=1.4,
                     loc="upper left", bbox_to_anchor=(0.578, 0.89))
    _frame_legend(leg)

    panel_labels = [(ax_a, "(a)", -0.32, 0.93), (ax_b, "(b)", -0.3, 0.88), (ax_c, "(c)", -0.3, 0.86)]
    for ax, lab, x, y in panel_labels:
        ax.text(x, y, lab, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")

    _frame_axes(ax_a, ax_b, ax_c)


# ===========================================
# ============== Figure 4 ===================
# ===========================================

def setup_fig4_format():
    _apply_style(14)
    fig, axes = plt.subplots(1, 2, figsize=(8*0.8,4*0.8))
    ax_a, ax_b = axes

    colors = ['darkmagenta', 'orchid', 'mediumblue', 'skyblue']
    styles = {'populations': [dict(color=c, linewidth=2) for c in colors],
              'L2': dict(color='coral', linewidth=2)}

    return fig, (ax_a, ax_b), styles


def finalize_fig4_format(fig, axes):
    ax_a, ax_b = axes

    ax_a.set_xlim(0, 2*np.pi)
    ax_a.set_xticks([0, np.pi, 2*np.pi])
    ax_a.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax_a.set_xlabel(r'$\Omega_0 t$')
    ax_a.set_ylabel(r'$P_{N,l}(t)$')

    leg = ax_a.legend(frameon=True, handletextpad=0.5, borderpad=0.3, handlelength=0.6, loc=7, title=r'$|N,l,m\!=\!0\rangle$')
    _frame_legend(leg)

    ax_b.set_xlim(0, 10*np.pi)
    ax_b.set_xticks([0, 5*np.pi, 10*np.pi])
    ax_b.set_xticklabels(['0', r'$5\pi$', r'$10\pi$'])
    ax_b.set_xlabel(r'$\Omega_0 t$')
    ax_b.set_ylabel("$\\langle L^2\\rangle$")

    _frame_axes(ax_a, ax_b)

    panel_labels = [(ax_a, "(a)", -0.28, 0.93), (ax_b, "(b)", -0.28, 0.93)]
    for ax, lab, x, y in panel_labels:
        ax.text(x, y, lab, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")

    plt.subplots_adjust(wspace=0.35)


# ===========================================
# ============== Figure 5 ===================
# ===========================================

def setup_fig5b_format():
    _apply_style(20)
    fig, ax_b = plt.subplots(1, 1, figsize=(4.4, FIG5_HEIGHT), dpi=DPI_2X)

    # the 'scattering' line is only a legend entry: it is drawn exactly under the bound state
    styles = {'spectrum':   dict(color='gray', alpha=0.1),
              'scattering': dict(color='gray', label='Scattering'),
              'bound':      dict(color='blue', label='Bound')}

    return fig, ax_b, styles


def finalize_fig5b_format(fig, ax_b):
    ax_b.set_xticks([0,np.pi,2*np.pi], labels=["0", r"$\pi$", r"2$\pi$"])
    ax_b.set_xlabel(r'$K$')
    ax_b.set_ylabel(r'$EU/J^2$')
    ax_b.set_xlim(0,2*np.pi)

    leg = ax_b.legend(frameon=True, handletextpad=0.5, borderpad=0.5, handlelength=1, loc=9)
    _frame_legend(leg)

    _frame_axes(ax_b)


def setup_fig5c_format(ps):
    _apply_style(20)
    base = plt.colormaps.get_cmap('Blues')
    colors = base(np.linspace(0, 1, len(ps)))
    colors[:, 2] = np.clip(colors[:, 2] + 0.5, 0, 1)
    blue_cmap = ListedColormap(colors)
    fig, ax_c = plt.subplots(1, 1, figsize=(3.0, FIG5_HEIGHT), dpi=DPI_2X)

    # one line per K, coloured by K
    styles = {'cmap': blue_cmap,
              'lines': [dict(color=blue_cmap(c), alpha=1, linewidth=3) for c in np.linspace(0, 1, len(ps))]}

    return fig, ax_c, styles


def finalize_fig5c_format(fig, ax_c, ps, styles):
    ax_c.set_xlim([0,5])
    #ax_c.set_ylim([-0.05,0.55])
    ax_c.set_xticks([0,1,2,3,4,5])
    ax_c.set_yticks([0,0.25,0.5, 0.75, 1])
    ax_c.set_xlabel('$R$')

    sm = plt.cm.ScalarMappable(cmap=styles['cmap'], norm=plt.Normalize(vmin=min(ps), vmax=max(ps)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_c)
    cbar.set_label(r'$K$', rotation=0, labelpad=0, fontsize=20)
    cbar.set_ticks([0,np.pi])
    cbar.set_ticklabels([0,R"$\pi$"])
    ax_c.set_ylabel(r'$|\psi_{K}|^2$', labelpad=10)

    _frame_axes(ax_c)

    plt.subplots_adjust(wspace=0.5)


# ===========================================
# ============== Figure 6 ===================
# ===========================================

def setup_fig6_format():
    _apply_style(20)
    fig, ax = plt.subplots(1, 1, figsize=(8,5), dpi=DPI_2X)
    axins = ax.inset_axes([0.62, 0.35, 0.35, 0.6])

    # one colour per glueball number; (main) grouped bars, (inset) smooth line with the points on top
    colors = ['seagreen','blueviolet','royalblue','coral']
    width = 0.2
    styles = {'colors': colors,
              'bars': [dict(width=width, color=c) for c in colors],
              'bar_offsets': [-1.5*width, -width/2, width/2, 1.5*width],
              'G2_lines': [dict(linewidth=2, color=c, zorder=2) for c in colors],
              'G2_markers': [dict(linestyle='none', zorder=3, color=c, marker='o', markersize=5) for c in colors]}

    return fig, (ax, axins), styles


def finalize_fig6_format(fig, axes, styles, ngs, nconfigs):
    ax, axins = axes

    ax.set_xticks(np.arange(nconfigs))
    ax.set_xticklabels([])
    ax.set_ylabel(r"$P_{\mathrm{config}}/P_{\mathrm{max}}$", labelpad=10)
    ax.set_ylim(0, 1.05)

    axins.set_xticks([0,2,4,6,8])
    axins.set_ylabel('$\\displaystyle\\frac{G_2(r)}{\\mathcal{N}(\\mathcal{N}-1)}$')
    axins.set_xlabel('$R$')
    handles = [Patch(facecolor=c, edgecolor='none') for c in styles['colors']]
    labels = ['$\\mathcal{N}$'+f'$={ng}$' for ng in ngs]
    axins.legend(handles, labels, loc=[0.43,0.35], frameon=False, handlelength=0.7, handleheight=0.7,
                 handletextpad=0.4, borderpad=0.2, labelspacing=0.3)

    _frame_axes(ax, axins)

    plt.subplots_adjust(wspace=0.4)
