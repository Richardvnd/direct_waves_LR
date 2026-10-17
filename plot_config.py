"""Figure sizes, colours and the matplotlib style shared by every script."""

import matplotlib.pyplot as plt
import numpy as np

COLOR_TOP = 0.95            # stop short of viridis's yellow end so the top colour stays visible


class PlotConfig:
    def __init__(self):
        self.style = "stylesheet.mplstyle"
        self.fig_width = 246.0 / 72.27
        self.fig_width_2 = 510.0 / 72.27
        self.fig_height = self.fig_width / 1.618
        self.fig_height_2 = self.fig_width_2 / 1.618
        self.colormap = plt.cm.viridis
        self.colors = [self.colormap(COLOR_TOP * x**0.5) for x in np.linspace(0, 1, 8)]
        self.colormap2 = plt.cm.magma
        self.color_bounce = "#991946"
        self.color_lr = "#c47a00"

    def colors2(self, n):
        """`n` magma colours, trimmed of the extremes."""
        return [self.colormap2(x) for x in np.linspace(0.1, 0.85, n)]

    def apply_style(self):
        plt.style.use(self.style)
