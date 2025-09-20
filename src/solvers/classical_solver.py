from typing import List
import numpy as np
from src.models.solver_state import SolverState
from .base_solver import BaseSolver

class LineSolver:

    def _generate_valid_solutions(self, length: int, constraint: List[int]) -> List[np.ndarray]:
        """Generate all possible valid arrangements"""
        if not constraint:
            return [np.zeros(length, dtype=int)]
        
        solutions = []
        min_length = sum(constraint) + len(constraint) - 1
        
        if min_length > length:
            return []
    
        def place_blocks(pos: int, block_idx: int, current: np.ndarray):
            if block_idx >= len(constraint):
                solutions.append(current.copy())
                return
            
            block_size = constraint[block_idx]
            max_start = length - sum(constraint[block_idx:]) - (len(constraint) - block_idx - 1)
            
            for start in range(pos, max_start + 1):
                new_current = current.copy()
                new_current[start:start + block_size] = 1
                next_pos = start + block_size + 1
                place_blocks(next_pos, block_idx + 1, new_current)
        
        place_blocks(0, 0, np.zeros(length, dtype=int))
        return solutions

    def _is_compatible(self, line: np.ndarray, solution: np.ndarray) -> bool:
        """Check compatibility between partial and complete solution"""
        for i in range(len(line)):
            if line[i] != -1 and line[i] != solution[i]:
                return False
        return True

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