from typing import Any, Dict, List, Sequence, Tuple
from bpp1d.structure import BinSolution, BppBin, Solution
from bpp1d.utils.heuristic_choice import HeuristicChoiceFn, best_fit_choice
from .model import Model, ModelStatus

class HeuristicModel(Model):


    def __init__(self, capacity: int, instance: Sequence[int], name: str = 'model',
                    choice_fn: HeuristicChoiceFn | None = None, offline_desc=False, *args, **kwargs):
        super().__init__(capacity, instance, name)
        self.choice_fn = choice_fn if choice_fn is not None else best_fit_choice
        self.bins: List[BppBin] = []
        self.offline_desc = offline_desc

    def solve(self) -> Tuple[Solution, Dict | None]:
        self.status = ModelStatus.SOLVING

        if self.offline_desc:
            self.instance = sorted(self.instance, reverse=True)

        for item in self.instance:
            choice = self.choice_fn(item, self.bins)
            if choice < 0:
                self.bins.append(BppBin(self.capacity, [item]))
            else:
                self.bins[choice].pack(item)
        
        self.status = ModelStatus.FINISHED
        return BinSolution(self.capacity, self.bins), None


class HarmonicKModel(HeuristicModel):
    def __init__(self, capacity: int, instance: Sequence[int], name: str = 'hamonic-k', K: int = 20):
        super().__init__(capacity, instance, name, None)
        # self.bin_types = []
        self.bin_with_types: List[Tuple[BppBin, int]] = []

        self.K = K

    def build(self) -> Any:
        self.intervals =  [(1/(j+1), 1/j) for j in range(1, self.K)]
        self.intervals.append((0, 1/self.K))
        return super().build()

    def get_interval(self, item: int):
        i_type = self.K
        for i  in range(self.K):
            j_1, j_0 = self.intervals[i]
            if j_1 < item / self.capacity <= j_0:
                i_type = i+1
                break
        return i_type

    def find_with_type(self, item: int, target_type: int):
        for bin, bin_type in self.bin_with_types:
            if bin.empty_space < item:
                continue

            if target_type == self.K:
                if bin_type == self.K:
                    return bin
            elif target_type == bin_type:
                if len(bin) < target_type:
                    return bin

        return None

    def solve(self) -> Tuple[Solution, Dict | None]:
        for item in self.instance:
            i_type = self.get_interval(item)
            candidate_bin = self.find_with_type(item, i_type)
            # if i_type < self.K:
            # found = False
            # for b, b_type in self.bin_with_types:
            #     if b_type != i_type or b.empty_space < item:
            #         continue
            #     # count_same_piece = len([ib for ib in b if self.get_interval(ib) == i_type])
            #     if i_type == self.K or (len(b) <= i_type and i_type != self.K):
            #         found = True
            #         b.pack(item)
                    # self.bin_types.append(i_type)
                    # break
            if not candidate_bin:
                self.bin_with_types.append((BppBin(self.capacity, [item]), i_type))
            else:
                candidate_bin.pack(item)
        self.bins = [b[0] for b in self.bin_with_types]
        self.status = ModelStatus.FINISHED

        return BinSolution(self.capacity, self.bins), {}


# class RefinedHarmonicModel(HeuristicModel):
#     def __init__(self, capacity: int, instance: Sequence[int], name: str = 'hamonic-k', K: int = 20):
#         super().__init__(capacity, instance, name, None)
#         self.bin_types = []

#         self.K = K

#     def build(self) -> Any:
#         self.intervals_1 = (59/96, 1)
#         self.intervals_a = (1/2, 59/96)
#         self.intervals_2 = (37/96, 1/2)
#         self.intervals_b = (1/3, 37/96)
#         self.intervals =  [(1/(j+1), 1/j) for j in range(3, self.K)]
#         self.intervals.append((0, 1/self.K))
#         return super().build()

#     def get_interval(self, item: int):
#         if self.intervals_1[0] < item  <= self.intervals_1[1]:
#             return 1
#         elif self.intervals_a[0] < item  <= self.intervals_a[1]:
#             return 'a'
#         elif self.intervals_2[0] < item  <= self.intervals_2[1]:
#             return 2
#         elif self.intervals_b[0] < item  <= self.intervals_b[1]:
#             return 'b'

#         i_type = self.K
#         for i  in range(self.K):
#             j_1, j_0 = self.intervals[i]
#             if j_1 < item / self.capacity <= j_0:
#                 i_type = i+3
#                 break
#         return i_type
    
#     def harmonic_pack(self, item : int):
#         i_type = self.get_interval(item)
#         bin = self.find_bin_with_type(item, i_type)
#         if bin is None:
#             self.bins.append(BppBin(self.capacity, [item]))
#             self.bin_types.append(i_type)
#         else:
#             bin.pack(item)

#     def find_bin_with_type(self, item, t):
#         for bin, b_type in zip(self.bins, self.bin_types):
#             if bin.empty_space > item:
#                 continue
#             if self.get_interval(item) == self.K or isinstance(self.get_interval(item), str):
#                 if b_type == t:
#                     return bin
#             elif isinstance(self.get_interval(item), int):
#                 if len(bin) < self.get_interval(item):
#                     return bin
                
#             # if b_type == t and self.get_interval(item) == self.K:
#                 # return bin
#             # elif 

#         return None


#     def solve(self) -> Tuple[Solution, Dict | None]:
#         N_a = N_b = N_ab = N_bb = N_b = N_c = 0
#         for item in self.instance:
#             i_type = self.get_interval(item)
#             if i_type <= self.K:
#                 self.harmonic_pack(item, i_type)

#             elif i_type == 'a':
#                 if N_b != 1:
#                     bin = self.find_bin_with_type('ab')
#                     if not bin:
#                         self.bins.append(BppBin(self.capacity, [item]))
#                         self.bin_types.append('ab')
#                         N_a += 1
#                     else:
#                         bin.append(item)
#                         N_b -= 1
#                         self.bins.append(BppBin(self.capacity, [item]))
#                         self.bin_types.append('b')
#                     else:
#                         bin.append(item)
#                     N_b = 0
#                     N_bb += 1
#                 elif N_bb <= 3 * N_c:
                    

                    

#         self.status = ModelStatus.FINISHED
#         return BinSolution(self.capacity, self.bins), {}
#                         N_ab += 1
#             elif i_type == 'b':
#                 if N_b == 1:
#                     bin = self.find_bin_with_type('b')
#                     if not bin:
#                         self.bins.append(BppBin(self.capacity, [item]))
#                         self.bin_types.append('b')
#                     else:
#                         bin.append(item)
#                     N_b = 0
#                     N_bb += 1
#                 elif N_bb <= 3 * N_c:


                    

#         self.status = ModelStatus.FINISHED
#         return BinSolution(self.capacity, self.bins), {}