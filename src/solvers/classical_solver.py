from typing import List
import numpy as np
from src.models.solver_state import SolverState
from .base_solver import BaseSolver

class LineSolver:
    def solve_line(self, line: np.ndarray, constraint: List[int]) -> bool:
        """Improved line solving with dynamic programming"""
        if not constraint:  # Empty constraint
            changed = np.any(line == -1)
            line[line == -1] = 0
            return changed
            
        possible_solutions = self._generate_valid_solutions(len(line), constraint)
        compatible_solutions = [sol for sol in possible_solutions 
                              if self._is_compatible(line, sol)]
        
        if not compatible_solutions:
            return False
            
        # Find definite cells
        changed = False
        for i in range(len(line)):
            if line[i] == -1:
                values = {sol[i] for sol in compatible_solutions}
                if len(values) == 1:
                    line[i] = values.pop()
                    changed = True
                    
        return changed

class ClassicalSolver(BaseSolver):
    def __init__(self, state: SolverState):
        super().__init__(state)
        self.line_solver = LineSolver()
    
    def solve_step(self) -> bool:
        """Perform one solving iteration"""
        return self.propagate_constraints()
    
    def solve(self, max_iterations: int = 100) -> bool:
        """Solve using classical constraint propagation"""
        for iteration in range(max_iterations):
            self.state.iteration = iteration
            
            if not self.solve_step():
                break  # No more progress possible
            
            if self.is_solved:
                return True
        
        return self.is_solved
    
    def propagate_constraints(self) -> bool:
        """Returns True if progress was made"""
        progress = False
        
        # Solve rows
        for i, constraint in enumerate(self.state.row_constraints):
            line = self.state.grid[i, :].copy()
            if self.line_solver.solve_line(line, constraint):
                self.state.grid[i, :] = line
                progress = True
        
        # Solve columns  
        for j, constraint in enumerate(self.state.col_constraints):
            line = self.state.grid[:, j].copy()
            if self.line_solver.solve_line(line, constraint):
                self.state.grid[:, j] = line
                progress = True
        
        return progress