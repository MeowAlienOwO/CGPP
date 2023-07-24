import matplotlib.pyplot as plt
from os import path
from matplotlib.patches import Rectangle
from bpp1d.structure import BppBin


class Visualizer:

    def __init__(self, visualize_dir) -> None:
        self.visualize_dir = visualize_dir
        self.colors = ['#8abcd1', '#2f90b9', '#1772b4']
        self.item_color = '#1772b4'
        self.empty_color = '#ee3f4d'

    def visualize_solutions(self, file_name, title, solutions, subtitles=[], item_range = (10, 60)):

        bin_height = 5
        rect_split_height = 0.1
        bin_split_width=0.2
        rect_width = 0.5

        n_solutions = len(solutions)

        fig, axs = plt.subplots(n_solutions, 1, constrained_layout=True)
        fig.suptitle(title)
        for i, solution in enumerate(solutions):
            ax = axs[i]
            if subtitles:
                ax.set_title(subtitles[i])
            ax.set_xlim([-0.5-bin_split_width,  len(solution) * (rect_width +bin_split_width) + 0.5])
            ax.set_xticks([])
            ax.set_ylim([0, 1.2 * bin_height])
            ax.set_yticks([])

            min_item, max_item = (
                min([item_range[0] for b in solution]),
                max([item_range[1] for b in solution])
            )
            color_step = (max_item - min_item) / len(self.colors)
            bin_bl = [0, 0]

            for bin in solution:
                # print(solution)
                rect_bl = bin_bl.copy()

                # bin = sorted(bin)
                items = bin if isinstance(bin, BppBin) else bin['items']
                

                for item in items:
                    i_color = min(int((item - min_item) // color_step), len(self.colors) - 1)
                    rect_height = bin_height * (item / bin.capacity)
                    ax.add_patch(Rectangle(tuple(rect_bl), rect_width, rect_height, color=self.colors[i_color]))
                    rect_bl[1] += (rect_height + rect_split_height)

                if solution.capacity - sum(bin) > 0:
                    rect_height = bin_height *((solution.capacity - sum(bin)) /solution.capacity) - rect_split_height
                    ax.add_patch(Rectangle(tuple(rect_bl), rect_width, rect_height, color='#ee3f4d'))

                bin_bl[0] += rect_width + bin_split_width
        
        plt.savefig(path.join(self.visualize_dir, file_name))
        plt.close()

    def visualize_solution(self, file_name, title, solution, item_range = (10, 60)):
        bin_height = 5
        rect_split_height = 0.1
        bin_split_width=0.2
        rect_width = 0.5
        fig, axs = plt.subplots(1, 1, constrained_layout=True)

        axs.set_xlim([-0.5-bin_split_width,  len(solution) * (rect_width +bin_split_width) + 0.5])
        axs.set_xticks([])
        axs.set_ylim([0, 1.2 * bin_height])
        axs.set_yticks([])

        min_item, max_item = item_range

        color_step = (max_item - min_item) / len(self.colors)
        bin_bl = [0, 0]

        for bin in solution:
            # print(solution)
            rect_bl = bin_bl.copy()



            for item in bin:
                i_color = min(int((item - min_item) // color_step), len(self.colors) - 1)
                rect_height = bin_height * (item / bin.capacity)
                axs.add_patch(Rectangle(tuple(rect_bl), rect_width, rect_height, color=self.item_color))
                rect_bl[1] += (rect_height + rect_split_height)

            if bin.capacity - sum(bin) > 0:
                rect_height = bin_height *((solution.capacity - sum(bin)) /solution.capacity) - rect_split_height
                axs.add_patch(Rectangle(tuple(rect_bl), rect_width, rect_height, color='#ee3f4d'))

            bin_bl[0] += rect_width + bin_split_width
        
        plt.savefig(path.join(self.visualize_dir, file_name))
        plt.close()

