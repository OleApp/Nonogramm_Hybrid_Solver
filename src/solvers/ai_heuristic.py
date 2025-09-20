import numpy as np
from typing import List, Tuple, Optional
from ..models.solver_state import SolverState
from .base_solver import BaseSolver

class AIHeuristic(BaseSolver):
    """AI-based heuristic solver for nonograms"""
    
    def __init__(self, state: SolverState):
        super().__init__(state)
        self.confidence_threshold = 0.7
    
    def solve_step(self) -> bool:
        """Perform one AI-guided solving step"""
        # Get cell predictions with confidence scores
        predictions = self._predict_cells()
        
        if not predictions:
            return False
        
        # Apply high-confidence predictions
        progress = False
        for row, col, value, confidence in predictions:
            if confidence > self.confidence_threshold:
                if self.state.grid[row, col] == -1:  # Only change unknown cells
                    self.state.grid[row, col] = value
                    self.state.certainty_map[row, col] = confidence
                    progress = True
        
        return progress
    
    def solve(self, max_iterations: int = 100) -> bool:
        """Solve using AI heuristics"""
        for iteration in range(max_iterations):
            self.state.iteration = iteration
            
            if not self.solve_step():
                break
            
            if self.is_solved:
                return True
        
        return self.is_solved
    
    def _predict_cells(self) -> List[Tuple[int, int, int, float]]:
        """
        Predict cell values using AI model
        Returns: List of (row, col, value, confidence) tuples
        """
        predictions = []
        
        # TODO: Replace with actual neural network inference
        # For now, use simple heuristics
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if self.state.grid[i, j] == -1:  # Unknown cell
                    prediction = self._heuristic_prediction(i, j)
                    if prediction:
                        predictions.append(prediction)
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x[3], reverse=True)
        return predictions[:10]  # Return top 10 predictions
    
    def _heuristic_prediction(self, row: int, col: int) -> Optional[Tuple[int, int, int, float]]:
        """
        Simple heuristic-based prediction for a cell
        Returns: (row, col, predicted_value, confidence) or None
        """
        # Analyze surrounding cells
        neighbors_filled = self._count_neighbors(row, col, value=1)
        neighbors_empty = self._count_neighbors(row, col, value=0)
        
        # Simple rule: if mostly surrounded by filled cells, predict filled
        total_neighbors = neighbors_filled + neighbors_empty
        if total_neighbors > 0:
            fill_ratio = neighbors_filled / total_neighbors
            if fill_ratio > 0.7:
                return (row, col, 1, min(0.8, fill_ratio))
            elif fill_ratio < 0.3:
                return (row, col, 0, min(0.8, 1.0 - fill_ratio))
        
        return None
    
    def _count_neighbors(self, row: int, col: int, value: int) -> int:
        """Count neighboring cells with specific value"""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.state.shape[0] and 
                    0 <= nc < self.state.shape[1] and
                    self.state.grid[nr, nc] == value):
                    count += 1
        return count
    
    def get_uncertainty_map(self) -> np.ndarray:
        """Return uncertainty map for visualization"""
        uncertainty = np.ones_like(self.state.grid, dtype=float)
        uncertainty[self.state.grid != -1] = 0.0  # Known cells have no uncertainty
        
        # Add prediction uncertainty for unknown cells
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if self.state.grid[i, j] == -1:
                    prediction = self._heuristic_prediction(i, j)
                    if prediction:
                        uncertainty[i, j] = 1.0 - prediction[3]  # Invert confidence
        
        return uncertainty
    
    def set_confidence_threshold(self, threshold: float):
        """Adjust confidence threshold for predictions"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))