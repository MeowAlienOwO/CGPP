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
                    figsize=(14, 14),
                    fontsize=18) -> None:
        self.visualize_dir = visualize_dir
        self.colors = colors
        self.empty_color = empty_color
        self.bin_height = bin_height
        self.rect_split_height = rect_split_height
        self.bin_split_width = bin_split_width
        self.rect_width = rect_width
        self.figsize = figsize
        self.fontsize = fontsize

        self.indices_size = 10
        self.fill_levels=[0.7, 0.8, 0.9, 0.95 ]

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
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xlim([0, max_len])
        # filled = np.array([b.filled_space for b in solution])

                            
        # wasted = np.array([b.empty_space for b in solution])
        # filled = np.pad(np.array([b.filled_space for b in solution]),
        #                     (0, max_len - solution.num_bins), mode='constant', constant_values= 0)
        # wasted = np.pad(np.array([b.empty_space for b in solution]),
        #                 (0, max_len - solution.num_bins), mode='constant', constant_values = 0)
        # bins = np.array(range(1, max_len + 1))


        # indices = list(range(self.indices_size))
        # section_size = int(np.ceil(len(solution) // self.indices_size))
        # # print(indices, section_size)
        # filled_sections = np.array([np.average(filled[i * section_size:max((i+i) * section_size, len(filled))]) 
        #                     for i in indices]) / solution.capacity
        # # wasted_sections = [np.average(wasted[i * section_size:max((i+i) * section_size, len(filled))])
        #                     # for i in indices]
        
        # print(filled_sections)

        # indices = list(range(self.indices_size))
        section_size = int(np.ceil(len(solution) // self.indices_size))

        filled_rate = {"rate": [],  "section": [], "level": []}
        for i, b in  enumerate(solution):
            section = (i / section_size) / self.indices_size
            for level in self.fill_levels:
                rate= b.filled_space / solution.capacity
                if rate > level:
                    filled_rate['rate'].append(rate)
                    filled_rate['section'].append(section)
                    filled_rate['level'].append(level)




        # for i in indices:
        #     section = filled[i * section_size:min((i+1) * section_size, len(filled))]
        #     # print(f"left: {i * section_size}, right: {min((i+1) * section_size, len(filled))}")
        #     # print(f'{section_size}')
        #     # print(f"len section {len(section)}")
        #     if(len(section) < 3):
        #         continue
        #     rates = self._count_levels(section, solution.capacity)
        #     for b in section:
            # print(rates)
        #     for level, rate in zip(self.fill_levels, rates):

        #         filled_rate['rate'].append(rate)
        #         filled_rate['level'].append(f"{level :.0%}")
        #         filled_rate['section'].append(i)
        data = pd.DataFrame.from_dict(filled_rate)
        # print(data)
        sns.histplot(ax=ax, data=data, x='section', hue='level' ,palette="Blues")
        if subtitle:
            ax.set_title(subtitle)


        # ax.bar(bins, filled, label='filled', color='#1772b4', width=self.rect_width)
        # ax.bar(bins, wasted, label='wasted', color='#ee3f4d',width=self.rect_width,  bottom=filled )

        # ax.bar(indices, filled_sections, label='filled', color='#1772b4')
        # ax.bar(indices, 1 - filled_sections, label='wasted', color='#ee3f4d',
        #         bottom=filled_sections )

    def add_solution_potentials_full(self, ax, solution: PotentialSolution,
                                        subtitle: str | None = None, max_len: int | None = None):
        if not max_len:
            max_len = solution.num_bins
        filled = [ solution.capacity for _ in range(solution.filled_bins)]
        wasted = [0 for _ in range(solution.filled_bins)]
        # ax.set_xticks([])
        # ax.set_ylim([0, 1.2 * self.bin_height])
        # ax.set_yticks([])
        # ax.set_xlim([0, max_len])
        
        for waste, num in enumerate(solution.potential[::-1]):
            filled += [solution.capacity - waste for _ in range(num)]
            wasted += [waste for _ in range(num)]

        # indices = list(range(self.indices_size))
        section_size = int(np.ceil(len(solution) // self.indices_size))

        filled_rate = {"rate": [],  "section": [], "level": []}
        for i, f in  enumerate(filled):
            section = (i / section_size) / self.indices_size
            for level in self.fill_levels:
                rate= f / solution.capacity
                if rate > level:
                    filled_rate['rate'].append(rate)
                    filled_rate['section'].append(section)
                    filled_rate['level'].append(level)


        # filled_rate = {"rate": [],  "section": [], "level": []}
        # for i in indices:
        #     section = filled[i * section_size:min((i+1) * section_size, len(filled))]
        #     if(len(section) < 3):
        #         continue
        #     rates = self._count_levels(section, solution.capacity)
        #     # print(rates)
        #     for level, rate in zip(self.fill_levels, rates):
        #         filled_rate['rate'].append(rate)
        #         filled_rate['level'].append(f"{level :.0%}")
        #         filled_rate['section'].append(i)
                # filled_rate['section'].append(f"{i * 1 / len(indices):.0%}")
        data = pd.DataFrame.from_dict(filled_rate)
        sns.histplot(ax=ax, data=data, x='section', hue='level', element='step', palette="Blues")


        # print(indices, section_size)
        # filled_sections = np.array([np.average(filled[i * section_size:max((i+i) * section_size, len(filled))]) 
        #                     for i in indices]) / solution.capacity
        # print(filled_sections)

        # filled = np.pad(np.array(filled),
        #                 (0, max_len - solution.num_bins), mode='constant')
        # wasted = np.pad(np.array(wasted),
        #                 (0, max_len - solution.num_bins), mode='constant')
        # bins = np.array(range(1, max_len + 1))
        if subtitle:
            # ax.set_title(subtitle)
            ax.set_title(subtitle)

        # ax.bar(bins, filled, label='filled', color='#1772b4', width=self.rect_width)
        # ax.bar(bins, wasted, label='wasted', color='#ee3f4d',width=self.rect_width,  bottom=filled )


        # ax.bar(indices, filled_sections, label='filled', color='#1772b4')
        # ax.bar(indices, 1 - filled_sections, label='wasted', color='#ee3f4d',
        #         bottom=filled_sections )
    # def visualize_bin_solution(self, file_name: str, title: str, solution:BinSolution):

    def _count_levels(self, input_fill_sequence, capacity):
        if len(input_fill_sequence) == 0:
            return np.zeros_like(self.fill_levels)
        input_fill_sequence = np.array(input_fill_sequence)
        res = np.zeros_like(self.fill_levels)
        # res = {}
        for i, level in enumerate(self.fill_levels):
            # print(np.sum(input_fill_sequence >= level))
            res[i] = np.sum(input_fill_sequence /capacity >= level) / len(input_fill_sequence)

        return res


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
        fig, ax = plt.subplots(len(solutions), 1)
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
        # sns.set(font_scale=2)

        plt.savefig(path.join(self.visualize_dir, file_name))
        plt.close()

    def visualize_solutions_sns(self, file_name: str, title: str, solutions: List[Solution], subtitles: List[str]=[]):


        data = {"model":[], "rate": [],  "section": [], "level": []}
        for  solution, model in zip(solutions, subtitles):


            section_size = int(np.ceil(solution.num_bins // self.indices_size))

            if isinstance(solution, BinSolution):
                
                for i, b in  enumerate(solution):
                    section = i // section_size
                    for level_l, level_r in list(zip([0] + self.fill_levels, self.fill_levels + [1]))[::-1]:
                        rate= b.filled_space / solution.capacity
                        if level_l < rate <= level_r:
                            data['rate'].append(rate)
                            data['section'].append(section)
                            data['level'].append(f"{level_l}-{level_r}")             
                            data['model'].append(model)
                            break
            elif isinstance(solution, PotentialSolution):
                filled = [ solution.capacity for _ in range(solution.filled_bins)]
                for waste, num in enumerate(solution.potential[::-1]):
                    filled += [solution.capacity - waste for _ in range(num)]
                for i, f in enumerate(filled):
                    # section = (i / section_size) / self.indices_size
                    section =i //  section_size
                    # for level in self.fill_levels:
                    for level_l, level_r in list(zip([0] + self.fill_levels, self.fill_levels + [1]))[::-1]:
                        rate= f / solution.capacity
                        # if rate > level:
                        if level_l < rate <= level_r:
                            data['rate'].append(rate)
                            data['section'].append(section)
                            # data['level'].append(level)
                            data['level'].append(f"{level_l}-{level_r}")             
                            data['model'].append(model)
                            break


        data = pd.DataFrame.from_dict(data)

        sns.set(font_scale=2)
        # fig = sns.FacetGrid(data, col = 'model', col_wrap=2)
        # fig.map_dataframe(sns.histplot, x='section', hue='level', multiple='stack', palette='Blues')
        fig = sns.displot(data, x='section', col='model', col_wrap=2, hue='level', multiple='stack', palette='Blues_r')
        # fig.add_legend(loc="upper left", bbox_to_anchor=(1, 1))

        fig.savefig(path.join(self.visualize_dir, file_name))
