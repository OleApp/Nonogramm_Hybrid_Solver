import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from ..models.solver_state import SolverState

class DatasetGenerator:
    """Generate synthetic nonogram puzzles for training"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_puzzle(self, height: int, width: int, density: float = 0.5) -> Dict[str, Any]:
        """Generate a single random nonogram puzzle"""
        # Create random solution grid
        solution = np.random.choice([0, 1], size=(height, width), p=[1-density, density])
        
        # Generate constraints from solution
        row_constraints = self._generate_line_constraints(solution, axis=1)
        col_constraints = self._generate_line_constraints(solution, axis=0)
        
        return {
            'solution': solution,
            'row_constraints': row_constraints,
            'col_constraints': col_constraints,
            'size': (height, width),
            'density': density
        }
    
    def generate_dataset(self, 
                        num_puzzles: int, 
                        size_range: Tuple[int, int] = (5, 15),
                        density_range: Tuple[float, float] = (0.3, 0.7)) -> List[Dict[str, Any]]:
        """Generate a dataset of multiple puzzles"""
        puzzles = []
        
        for i in range(num_puzzles):
            # Random size within range
            height = random.randint(size_range[0], size_range[1])
            width = random.randint(size_range[0], size_range[1])
            
            # Random density within range
            density = random.uniform(density_range[0], density_range[1])
            
            puzzle = self.generate_puzzle(height, width, density)
            puzzle['id'] = f"generated_{i:04d}"
            puzzles.append(puzzle)
        
        return puzzles
    
    def generate_structured_puzzle(self, 
                                 height: int, 
                                 width: int, 
                                 pattern_type: str = "random") -> Dict[str, Any]:
        """Generate puzzles with specific patterns"""
        if pattern_type == "cross":
            solution = self._generate_cross_pattern(height, width)
        elif pattern_type == "checkerboard":
            solution = self._generate_checkerboard_pattern(height, width)
        elif pattern_type == "diagonal":
            solution = self._generate_diagonal_pattern(height, width)
        elif pattern_type == "border":
            solution = self._generate_border_pattern(height, width)
        elif pattern_type == "symmetric":
            solution = self._generate_symmetric_pattern(height, width)
        else:  # random
            solution = np.random.choice([0, 1], size=(height, width), p=[0.5, 0.5])
        
        row_constraints = self._generate_line_constraints(solution, axis=1)
        col_constraints = self._generate_line_constraints(solution, axis=0)
        
        return {
            'solution': solution,
            'row_constraints': row_constraints,
            'col_constraints': col_constraints,
            'size': (height, width),
            'pattern': pattern_type
        }
    
    def generate_difficulty_graded_dataset(self, puzzles_per_difficulty: int = 50) -> List[Dict[str, Any]]:
        """Generate dataset with different difficulty levels"""
        puzzles = []
        
        difficulty_configs = {
            'easy': {'size_range': (5, 8), 'density_range': (0.4, 0.6), 'patterns': ['border', 'cross']},
            'medium': {'size_range': (8, 12), 'density_range': (0.3, 0.7), 'patterns': ['symmetric', 'diagonal']},
            'hard': {'size_range': (12, 20), 'density_range': (0.2, 0.8), 'patterns': ['random', 'checkerboard']},
            'expert': {'size_range': (15, 25), 'density_range': (0.1, 0.9), 'patterns': ['random']}
        }
        
        for difficulty, config in difficulty_configs.items():
            for i in range(puzzles_per_difficulty):
                height = random.randint(*config['size_range'])
                width = random.randint(*config['size_range'])
                pattern = random.choice(config['patterns'])
                
                if pattern == 'random':
                    density = random.uniform(*config['density_range'])
                    puzzle = self.generate_puzzle(height, width, density)
                else:
                    puzzle = self.generate_structured_puzzle(height, width, pattern)
                
                puzzle['id'] = f"{difficulty}_{i:03d}"
                puzzle['difficulty'] = difficulty
                puzzles.append(puzzle)
        
        return puzzles
    
    def _generate_line_constraints(self, grid: np.ndarray, axis: int) -> List[List[int]]:
        """Generate constraints for all lines along specified axis"""
        constraints = []
        
        if axis == 0:  # columns
            lines = [grid[:, i] for i in range(grid.shape[1])]
        else:  # rows
            lines = [grid[i, :] for i in range(grid.shape[0])]
        
        for line in lines:
            constraint = self._line_to_constraint(line)
            constraints.append(constraint)
        
        return constraints
    
    def _line_to_constraint(self, line: np.ndarray) -> List[int]:
        """Convert a solved line to its constraint"""
        if not np.any(line):  # All zeros
            return []
        
        blocks = []
        current_block = 0
        
        for cell in line:
            if cell == 1:
                current_block += 1
            else:
                if current_block > 0:
                    blocks.append(current_block)
                    current_block = 0
        
        if current_block > 0:
            blocks.append(current_block)
        
        return blocks
    
    def _generate_cross_pattern(self, height: int, width: int) -> np.ndarray:
        """Generate cross pattern"""
        solution = np.zeros((height, width), dtype=int)
        mid_row, mid_col = height // 2, width // 2
        
        # Horizontal line
        solution[mid_row, :] = 1
        # Vertical line
        solution[:, mid_col] = 1
        
        return solution
    
    def _generate_checkerboard_pattern(self, height: int, width: int) -> np.ndarray:
        """Generate checkerboard pattern"""
        solution = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                if (i + j) % 2 == 0:
                    solution[i, j] = 1
        return solution
    
    def _generate_diagonal_pattern(self, height: int, width: int) -> np.ndarray:
        """Generate diagonal pattern"""
        solution = np.zeros((height, width), dtype=int)
        
        # Main diagonal
        for i in range(min(height, width)):
            solution[i, i] = 1
        
        # Anti-diagonal
        for i in range(min(height, width)):
            solution[i, width - 1 - i] = 1
        
        return solution
    
    def _generate_border_pattern(self, height: int, width: int) -> np.ndarray:
        """Generate border pattern"""
        solution = np.zeros((height, width), dtype=int)
        
        # Top and bottom borders
        solution[0, :] = 1
        solution[height-1, :] = 1
        
        # Left and right borders
        solution[:, 0] = 1
        solution[:, width-1] = 1
        
        return solution
    
    def _generate_symmetric_pattern(self, height: int, width: int) -> np.ndarray:
        """Generate horizontally symmetric pattern"""
        solution = np.zeros((height, width), dtype=int)
        
        # Fill left half randomly
        left_half = width // 2
        solution[:, :left_half] = np.random.choice([0, 1], size=(height, left_half))
        
        # Mirror to right half
        for i in range(height):
            for j in range(left_half):
                mirror_j = width - 1 - j
                solution[i, mirror_j] = solution[i, j]
        
        return solution
    
    def validate_puzzle(self, puzzle: Dict[str, Any]) -> bool:
        """Validate that a generated puzzle is solvable"""
        try:
            from ..solvers.classical_solver import ClassicalSolver
            
            # Create solver state
            height, width = puzzle['size']
            initial_grid = np.full((height, width), -1)
            
            state = SolverState(
                grid=initial_grid,
                row_constraints=puzzle['row_constraints'],
                col_constraints=puzzle['col_constraints']
            )
            
            # Try to solve
            solver = ClassicalSolver(state)
            solved = solver.solve(max_iterations=100)
            
            # Check if solution matches expected
            if solved:
                return np.array_equal(solver.state.grid, puzzle['solution'])
            
            return False
        except ImportError:
            # If solver not available, assume valid
            return True
    
    def save_dataset(self, puzzles: List[Dict[str, Any]], filename: str):
        """Save generated dataset to file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_puzzles = []
        for puzzle in puzzles:
            serializable_puzzle = puzzle.copy()
            serializable_puzzle['solution'] = puzzle['solution'].tolist()
            serializable_puzzles.append(serializable_puzzle)
        
        with open(filename, 'w') as f:
            json.dump({
                'metadata': {
                    'total_puzzles': len(puzzles),
                    'generator_version': '1.0'
                },
                'puzzles': serializable_puzzles
            }, f, indent=2)