import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SolverState:
    grid: np.ndarray  # -1=unknown, 0=empty, 1=filled
    row_constraints: List[List[int]]
    col_constraints: List[List[int]]
    iteration: int = 0
    certainty_map: Optional[np.ndarray] = None  # AI confidence scores
    
    def __post_init__(self):
        """Initialize certainty_map if not provided"""
        if self.certainty_map is None:
            self.certainty_map = np.zeros_like(self.grid, dtype=float)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return grid dimensions"""
        return self.grid.shape
    
    @property
    def is_complete(self) -> bool:
        """Check if grid has no unknown cells"""
        return not np.any(self.grid == -1)
    
    def copy(self) -> 'SolverState':
        """Create a deep copy of the state"""
        return SolverState(
            grid=self.grid.copy(),
            row_constraints=self.row_constraints.copy(),
            col_constraints=self.col_constraints.copy(),
            iteration=self.iteration,
            certainty_map=self.certainty_map.copy() if self.certainty_map is not None else None
        )