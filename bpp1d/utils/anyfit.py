from typing import Callable, Sequence
from bpp1d.structure import BppBin


HeuristicChoiceFn = Callable[[int, Sequence[BppBin]], int]

def best_fit_choice(item: int, bins: Sequence[BppBin]) -> int:
    """Return the choice using best fit heuristic

    Args:
        item (int): item to be packed
        bins (Iterable[BppBin]): existing bins

    Returns:
        int: choiced index of target bin, -1 if a new bin is opened
    """
    zipped = [(idx, b) for idx, b in enumerate(bins) 
                            if b.empty_space - item >= 0]
    if len(zipped) == 0:
        choice = -1
    else:
        choice = min(zipped, key=lambda z: z[1].empty_space - item)[0]
    return choice

def first_fit_choice(item: int, bins: Sequence[BppBin]) -> int:
    """Return the choice using best fit heuristic

    Args:
        item (int): item to be packed
        bins (Iterable[BppBin]): existing bins

    Returns:
        int: choiced index of target bin, -1 if a new bin is opened
    """
    fit = [b for b in bins if b.empty_space - item >= 0]
    if len(fit) == 0:
        choice = len(fit)
    else:
        choice = bins.index(fit[0])
    return choice
