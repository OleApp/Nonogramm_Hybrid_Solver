from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Any
from ..models.solver_state import SolverState

class BaseSolver(ABC):
    """Abstract base class for all nonogram solvers"""
    
    def __init__(self, state: SolverState):
        self.state = state
        self.move_callbacks: list[Callable] = []
    
    @abstractmethod
    def solve(self, max_iterations: int = 100) -> bool:
        """
        Solve the nonogram puzzle
        Returns True if completely solved, False otherwise
        """
        pass
    
    @abstractmethod
    def solve_step(self) -> bool:
        """
        Perform one solving step
        Returns True if progress was made, False otherwise
        """
        pass
    
    def add_move_callback(self, callback: Callable[[Any, SolverState], None]):
        """Add callback function to be called after each move"""
        self.move_callbacks.append(callback)
    
    def _notify_move(self, move: Any):
        """Notify all callbacks about a move"""
        for callback in self.move_callbacks:
            callback(move, self.state)
    
    def reset(self):
        """Reset solver to initial state"""
        self.state.grid.fill(-1)  # Reset to all unknown
        self.state.iteration = 0
        if self.state.certainty_map is not None:
            self.state.certainty_map.fill(0.0)
    
    @property
    def is_solved(self) -> bool:
        """Check if puzzle is completely solved"""
        return self.state.is_complete
    
    @property
    def progress_percentage(self) -> float:
        """Calculate solving progress as percentage"""
        total_cells = self.state.grid.size
        known_cells = np.sum(self.state.grid != -1)
        return (known_cells / total_cells) * 100.0 if total_cells > 0 else 0.0