import matplotlib.pyplot as plt

COLORS = [
    '#A9D6E5',
    '#89C2D9',
    '#61A5C2',
    '#468FAF',
    '#2C7DA0',
    '#2A6F97',
    '#014F86',
    '#012A4A'
]

KM_COLORS = COLORS
LINE_STYLES = ['-', '--', '-.', ':']

def apply_plot_style():
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'lines.linewidth': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
