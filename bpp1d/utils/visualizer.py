from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from os import path
from matplotlib.patches import Rectangle
from bpp1d.structure import BppBin, BinSolution, PotentialSolution, Solution
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_COLORS = sns.color_palette('Blues', 10)
# ['#8abcd1', '#2f90b9', '#1772b4']
class Visualizer:

    def __init__(self, visualize_dir: Path,
                    colors= DEFAULT_COLORS,
                    empty_color='#ee3f4d',
                    bin_height=3,
                    rect_split_height=0.1,
                    bin_split_width=0.5,
                    rect_width=1,
                    figsize=(20, 10),
                    fontsize=32) -> None:
        self.visualize_dir = visualize_dir
        self.colors = colors
        self.empty_color = empty_color
        self.bin_height = bin_height
        self.rect_split_height = rect_split_height
        self.bin_split_width = bin_split_width
        self.rect_width = rect_width
        self.figsize = figsize
        self.fontsize = fontsize

    def get_item_color(self, item, capacity=100):
        color_step = capacity // len(self.colors)
        color_index = max(item // color_step , len(self.colors) - 1)
        return self.colors[color_index]

    def add_solution_bins(self, ax, solution, subtitle=None):
        ax.set_xticks([])
        ax.set_ylim([0, 1.2 * self.bin_height])
        ax.set_yticks([])
        bin_bl = [0, 0]
        if subtitle:
            ax.set_title(subtitle)

        for bin in solution:
            rect_bl = bin_bl.copy()

            items = bin if isinstance(bin, BppBin) else bin['items']

            for item in items:
                color = self.get_item_color(item, solution.capacity)
                rect_height = self.bin_height * (item / bin.capacity)
                ax.add_patch(Rectangle(rect_bl, self.rect_width, rect_height, color=color))
                rect_bl[1] += (rect_height + self.rect_split_height)

            if solution.capacity - sum(bin) > 0:
                rect_height = self.bin_height *((solution.capacity - sum(bin)) /solution.capacity) \
                                - self.rect_split_height
                ax.add_patch(Rectangle(rect_bl, self.rect_width, rect_height, color='#ee3f4d'))

            bin_bl[0] += self.rect_width + self.bin_split_width

    def add_solution_bins_full(self, ax, solution:BinSolution, 
                                subtitle: str | None=None, max_len: int | None=None):
        if not max_len:
            max_len = solution.num_bins
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, max_len])
        filled = np.pad(np.array([b.filled_space for b in solution]),
                            (0, max_len - solution.num_bins), mode='constant', constant_values= 0)
        wasted = np.pad(np.array([b.empty_space for b in solution]),
                        (0, max_len - solution.num_bins), mode='constant', constant_values = 0)
        bins = np.array(range(1, max_len + 1))

        if subtitle:
            ax.set_title(subtitle, fontsize=self.fontsize)


        ax.bar(bins, filled, label='filled', color='#1772b4', width=self.rect_width)
        ax.bar(bins, wasted, label='wasted', color='#ee3f4d',width=self.rect_width,  bottom=filled )


    def add_solution_potentials_full(self, ax, solution: PotentialSolution,
                                        subtitle: str | None = None, max_len: int | None = None):
        if not max_len:
            max_len = solution.num_bins
        filled = [ solution.capacity for _ in range(solution.filled_bins)]
        wasted = [0 for _ in range(solution.filled_bins)]
        ax.set_xticks([])
        # ax.set_ylim([0, 1.2 * self.bin_height])
        ax.set_yticks([])
        ax.set_xlim([0, max_len])
        for waste, num in enumerate(solution.potential[::-1]):
            filled += [solution.capacity - waste for _ in range(num)]
            wasted += [waste for _ in range(num)]

        filled = np.pad(np.array(filled),
                        (0, max_len - solution.num_bins), mode='constant')
        wasted = np.pad(np.array(wasted),
                        (0, max_len - solution.num_bins), mode='constant')
        bins = np.array(range(1, max_len + 1))
        if subtitle:
            # ax.set_title(subtitle)
            ax.set_title(subtitle, fontsize=self.fontsize)

        ax.bar(bins, filled, label='filled', color='#1772b4', width=self.rect_width)
        ax.bar(bins, wasted, label='wasted', color='#ee3f4d',width=self.rect_width,  bottom=filled )


    # def visualize_bin_solution(self, file_name: str, title: str, solution:BinSolution):


    #     fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
    #     if title:
    #         fig.suptitle(title)
    #     self.add_solution_bins(ax, solution)
    #     # self.add_solution_bins_full(ax, solution)
        
    #     plt.savefig(path.join(self.visualize_dir, file_name))
    #     plt.close()

    # def visualize_bin_solutions(self, file_name: str, title: str, 
    #                             solutions: List[BinSolution], subtitles: List[str]=[]):
        
    #     fig, ax = plt.subplots(len(solutions), 1, constrained_layout=True, figsize=self.figsize)
    #     if title:
    #         fig.suptitle(title)
    #     for i, solution in enumerate(solutions):
    #         subtitle = subtitles[i] if i < len(subtitles) else None
    #         self.add_solution_bins(ax[i], solution, subtitle)
        
    #     plt.savefig(path.join(self.visualize_dir, file_name))
    #     plt.close()

    def visualize_solutions(self, file_name: str, title: str, solutions: List[Solution], subtitles: List[str]=[]):
        fig, ax = plt.subplots(len(solutions), 1, constrained_layout=True, figsize=self.figsize)
        if title:
            fig.suptitle(title, fontsize=self.fontsize)

        max_len = max([s.num_bins for s in solutions])

        for i, solution in enumerate(solutions):
            subtitle = subtitles[i] if i < len(subtitles) else None
            if isinstance(solution, BinSolution):
                self.add_solution_bins_full(ax[i], solution, subtitle, max_len)
            elif isinstance(solution, PotentialSolution):
                self.add_solution_potentials_full(ax[i], solution, subtitle, max_len)
        # plt.xlim([0, max_len])
        # plt.axis('tight')
        # plt.rcParams.update({'font.size': self.fontsize})
        # plt.legend(fontsize=self.fontsize)

        plt.savefig(path.join(self.visualize_dir, file_name))
        plt.close()

