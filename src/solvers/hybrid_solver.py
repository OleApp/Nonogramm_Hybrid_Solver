import numpy as np
from ..models.solver_state import SolverState
from .base_solver import BaseSolver
from .classical_solver import ClassicalSolver
from .ai_heuristic import AIHeuristic

class HybridNonogramSolver(BaseSolver):
    def __init__(self, row_constraints, col_constraints):
        # Create initial state
        initial_state = SolverState(
            grid=np.full((len(row_constraints), len(col_constraints[0])), -1),
            row_constraints=row_constraints,
            col_constraints=col_constraints
        )
        super().__init__(initial_state)
        
        # Initialize component solvers
        self.classical_solver = ClassicalSolver(self.state)
        self.ai_heuristic = AIHeuristic(self.state)
        self.use_ai = True
        
    def solve_step(self) -> bool:
        """Perform one hybrid solving step"""
        # Phase 1: Try classical constraint propagation
        classical_progress = self.classical_solver.solve_step()
        
        if classical_progress:
            return True
        
        # Phase 2: If classical solver stuck, use AI heuristic
        if self.use_ai and self.ai_heuristic:
            ai_progress = self.ai_heuristic.solve_step()
            if ai_progress:
                return True
        
        return False
    
    def solve(self, max_iterations: int = 100) -> bool:
        """Main hybrid solving loop"""
        for iteration in range(max_iterations):
            self.state.iteration = iteration
            
            if not self.solve_step():
                break  # No more progress possible
                
            if self.is_solved:
                return True
        
        return self.is_solved
    
    def solve_with_strategy(self, strategy: str = "hybrid") -> bool:
        """Solve using specific strategy"""
        if strategy == "classical":
            return self.classical_solver.solve()
        elif strategy == "ai":
            return self.ai_heuristic.solve()
        elif strategy == "hybrid":
            return self.solve()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def get_solving_stats(self) -> dict:
        """Return statistics about the solving process"""
        return {
            "iterations": self.state.iteration,
            "progress_percentage": self.progress_percentage,
            "cells_solved": np.sum(self.state.grid != -1),
            "total_cells": self.state.grid.size,
            "is_complete": self.is_solved
        }
    
    def get_uncertainty_regions(self) -> np.ndarray:
        """Get regions where AI is uncertain"""
        if self.ai_heuristic:
            return self.ai_heuristic.get_uncertainty_map()
        return np.zeros_like(self.state.grid)
    
    def enable_ai(self, enable: bool = True):
        """Enable or disable AI component"""
        self.use_ai = enable
    
    def set_ai_confidence_threshold(self, threshold: float):
        """Adjust AI confidence threshold"""
        if self.ai_heuristic:
            self.ai_heuristic.set_confidence_threshold(threshold)