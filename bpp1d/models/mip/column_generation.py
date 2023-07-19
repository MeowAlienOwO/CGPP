from typing import Dict, List
from pulp import LpVariable, lpSum, LpProblem, LpAffineExpression
from pulp import LpMaximize, LpMinimize, LpInteger, LpContinuous, PULP_CBC_CMD
# import pulp2mat
import numpy as np
# from scipy.optimize import milp
from enum import Enum
from itertools import chain

from bpp1d.structure import BinPattern

class MipSolverStatus(Enum):
    FINISHED = 0
    UNFINISHED = 1
    TIMEOUT = 2

class ColumnGeneration:
    """Column generation of cutting stock problem
    """

    def __init__(self, capacity: int, demands: Dict[int, int], verbose: int=0):
        self.capacity = capacity
        self.demands = demands
        self.num_items = len(demands)
        self.status: MipSolverStatus = MipSolverStatus.UNFINISHED
        self.verbose = verbose
        # initalize patterns

    @property
    def items(self):
        return self.demands.keys()

    # def _call_solver(self, problem):
    #     c, integrality, constraints, bounds = pulp2mat.convert_all(problem)
    #     result = milp(c, integrality=integrality, constraints=constraints, bounds=bounds)
    #     pulp2mat.decode_solution(result, problem)

    def _call_solver(self, problem, relax:bool = False):
        problem.solve(PULP_CBC_CMD(msg = False, mip=not relax))
        
    def solve(self, max_iter: int=1000) -> Dict[BinPattern, int] | None:

        patterns = np.zeros((len(self.demands), len(self.demands)), dtype=np.int64)
        for i, item in enumerate(self.demands):
            patterns[i, i] = self.capacity // item
            # patterns[i, i] = 1
        
        master_prob, x, dual = self._solve_master(patterns)

        for _ in range(max_iter):

            assert dual is not None
            dual_prob, delta = self._solve_dual(dual)
            new_pattern = np.array([delta[i].value() for i in range(self.num_items)])
            # print(new_pattern, dual_prob.objective.value())
            patterns = np.vstack((patterns, new_pattern))

            master_prob, x, dual = self._solve_master(patterns)
            
            if self.verbose > 0:
                print(master_prob.objective.values())

            if dual_prob.objective.value() <= 1 + 10e-7:
                self.status = MipSolverStatus.FINISHED
                break
        
        if self.status != MipSolverStatus.FINISHED:
            self.status = MipSolverStatus.TIMEOUT
            return None
        else:
            master_prob, x, _ = self._solve_master(patterns, relax=False)
            # master_prob, x, _ = self._solve_master(patterns, relax=True)
            plan = {
                self._decode_pattern(patterns[i]): x[i].value()
                for i in range(patterns.shape[0])
                if x[i].value() > 0
            }
            return plan


    def _decode_pattern(self, pattern: np.ndarray):
        items = [[item] * int(num) for num, item in zip(pattern, self.items)]
        return BinPattern(chain(*items))



    def _solve_master(self, patterns: np.ndarray, relax: bool = True):
        master_prob = LpProblem("MainProblem", LpMinimize)
        vartype = LpContinuous if relax else LpInteger
        num_patterns = patterns.shape[0]
        # print(patterns)
        # variables
        x = {
            j: LpVariable(f"x_{j}", lowBound=0, upBound=None, cat=vartype) 
            for j in range(num_patterns)
        }

        constraints: List[LpAffineExpression] = []
        # constraint
        for j, item in enumerate(self.items):
            c =  lpSum(patterns[i, j] * x[i] 
                        for i in range(num_patterns)) >= self.demands[item]
            constraints.append(c)
            master_prob += c


        # object
        master_prob += lpSum(x[i] for i in range(num_patterns))

        self._call_solver(master_prob, relax)
        duals = None
        if relax:
            duals = [c.pi for _, c in master_prob.constraints.items()]
        # for i in range(num_patterns):
        #     print(x[i].value(), patterns[i])
        
        # print([(v, v.value()) for v in x.values()])
        
        return master_prob, x, duals

        
    def _solve_dual(self, duals):
        
        dual_prob = LpProblem("DualProblem", LpMaximize)
        # variables
        delta = {
            i: LpVariable(f"delta_{i}", lowBound=0, upBound=None, cat=LpInteger)
            for i in range(self.num_items)
        }
        # print(duals)
        
        # constraint

        dual_prob += lpSum(delta[i] * item for i, item in enumerate(self.items)) <= self.capacity

        # object

        dual_prob += lpSum(delta[i] *duals[i] for i in range(self.num_items))

        self._call_solver(dual_prob)
        return dual_prob, delta







if __name__ == "__main__":
    instance = [5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]

    demands = {i: instance.count(i) for i in sorted(set(instance))}
    capacity = 10
    print(demands)

    cg = ColumnGeneration(capacity, demands)
    result = cg.solve()
    print(result)
