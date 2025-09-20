import numpy as np
from typing import List, Tuple, Optional
from models.solver_state import SolverState

class NonogramValidator:
    """Validation utilities for nonogram puzzles and solutions"""
    
    @staticmethod
    def is_valid_partial_solution(state: SolverState) -> bool:
        """Check if current partial solution doesn't violate constraints"""
        # Validate rows
        for i, constraint in enumerate(state.row_constraints):
            if not NonogramValidator._is_valid_partial_line(state.grid[i, :], constraint):
                return False
        
        # Validate columns
        for j, constraint in enumerate(state.col_constraints):
            if not NonogramValidator._is_valid_partial_line(state.grid[:, j], constraint):
                return False
        
        return True
    
    @staticmethod
    def is_complete_solution(state: SolverState) -> bool:
        """Check if solution is complete and valid"""
        # Must have no unknown cells
        if np.any(state.grid == -1):
            return False
        
        # Must satisfy all constraints
        return NonogramValidator.is_valid_partial_solution(state)
    
    @staticmethod
    def _is_valid_partial_line(line: np.ndarray, constraint: List[int]) -> bool:
        """Check if a partial line doesn't violate its constraint"""
        if not constraint:  # Empty constraint - line should be all empty
            return not np.any(line == 1)
        
        # Get current filled blocks
        filled_blocks = NonogramValidator._get_filled_blocks(line)
        unknown_positions = np.where(line == -1)[0]
        
        # If no unknowns, check exact match
        if len(unknown_positions) == 0:
            return filled_blocks == constraint
        
        # With unknowns, check if current blocks could lead to valid solution
        return NonogramValidator._could_be_valid(filled_blocks, constraint, len(unknown_positions))
    
    @staticmethod
    def _get_filled_blocks(line: np.ndarray) -> List[int]:
        """Extract current filled block sizes from line"""
        blocks = []
        current_block = 0
        in_unknown_region = False
        
        for cell in line:
            if cell == 1:  # Filled
                current_block += 1
                in_unknown_region = False
            elif cell == 0:  # Empty
                if current_block > 0 and not in_unknown_region:
                    blocks.append(current_block)
                    current_block = 0
                in_unknown_region = False
            else:  # Unknown (-1)
                in_unknown_region = True
        
        # Add final block if line ends with filled cells
        if current_block > 0 and not in_unknown_region:
            blocks.append(current_block)
        
        return blocks
    
    @staticmethod
    def _could_be_valid(current_blocks: List[int], constraint: List[int], unknowns: int) -> bool:
        """Check if current partial blocks could lead to valid solution"""
        if len(current_blocks) > len(constraint):
            return False
        
        # Check if current blocks match beginning of constraint
        for i, block in enumerate(current_blocks):
            if i >= len(constraint) or block > constraint[i]:
                return False
        
        # Estimate if remaining unknowns could satisfy remaining constraint
        remaining_constraint = constraint[len(current_blocks):]
        min_needed = sum(remaining_constraint) + len(remaining_constraint) - 1
        
        return min_needed <= unknowns
    
    @staticmethod
    def validate_constraints(row_constraints: List[List[int]], 
                           col_constraints: List[List[int]], 
                           grid_size: Tuple[int, int]) -> bool:
        """Validate that constraints are consistent with grid size"""
        height, width = grid_size
        
        # Check row constraints
        if len(row_constraints) != height:
            return False
        
        for i, constraint in enumerate(row_constraints):
            min_width = sum(constraint) + len(constraint) - 1 if constraint else 0
            if min_width > width:
                return False
        
        # Check column constraints
        if len(col_constraints) != width:
            return False
        
        for j, constraint in enumerate(col_constraints):
            min_height = sum(constraint) + len(constraint) - 1 if constraint else 0
            if min_height > height:
                return False
        
        return True
    
    @staticmethod
    def get_constraint_violations(state: SolverState) -> List[Tuple[str, int, str]]:
        """Get list of constraint violations with details"""
        violations = []
        
        # Check rows
        for i, constraint in enumerate(state.row_constraints):
            line = state.grid[i, :]
            if not NonogramValidator._is_valid_partial_line(line, constraint):
                violations.append(("row", i, f"Row {i} violates constraint {constraint}"))
        
        # Check columns
        for j, constraint in enumerate(state.col_constraints):
            line = state.grid[:, j]
            if not NonogramValidator._is_valid_partial_line(line, constraint):
                violations.append(("col", j, f"Column {j} violates constraint {constraint}"))
        
        return violations
    
    @staticmethod
    def calculate_consistency_score(state: SolverState) -> float:
        """Calculate how consistent current state is (0.0 to 1.0)"""
        total_lines = len(state.row_constraints) + len(state.col_constraints)
        valid_lines = 0
        
        # Check rows
        for i, constraint in enumerate(state.row_constraints):
            if NonogramValidator._is_valid_partial_line(state.grid[i, :], constraint):
                valid_lines += 1
        
        # Check columns
        for j, constraint in enumerate(state.col_constraints):
            if NonogramValidator._is_valid_partial_line(state.grid[:, j], constraint):
                valid_lines += 1
        
        return valid_lines / total_lines if total_lines > 0 else 1.0
    
    @staticmethod
    def find_definite_cells(line: np.ndarray, constraint: List[int]) -> np.ndarray:
        """Find cells that must be filled/empty based on constraint"""
        if not constraint:
            # Empty constraint - all unknowns must be empty
            definite = line.copy()
            definite[line == -1] = 0
            return definite
        
        # Generate all possible valid arrangements
        possible_solutions = NonogramValidator._generate_line_solutions(len(line), constraint)
        
        # Filter solutions compatible with current line
        compatible = [sol for sol in possible_solutions 
                     if NonogramValidator._is_line_compatible(line, sol)]
        
        if not compatible:
            return line  # No valid solutions found
        
        # Find definite cells (same in all compatible solutions)
        definite = line.copy()
        for i in range(len(line)):
            if line[i] == -1:  # Only update unknown cells
                values = {sol[i] for sol in compatible}
                if len(values) == 1:
                    definite[i] = values.pop()
        
        return definite
    
    @staticmethod
    def _generate_line_solutions(length: int, constraint: List[int]) -> List[np.ndarray]:
        """Generate all possible arrangements of blocks in line"""
        if not constraint:
            return [np.zeros(length, dtype=int)]
        
        solutions = []
        min_length = sum(constraint) + len(constraint) - 1
        
        if min_length > length:
            return []  # Impossible constraint
        
        def place_blocks(pos: int, block_idx: int, current: np.ndarray):
            if block_idx >= len(constraint):
                solutions.append(current.copy())
                return
            
            block_size = constraint[block_idx]
            max_start = length - sum(constraint[block_idx:]) - (len(constraint) - block_idx - 1)
            
            for start in range(pos, max_start + 1):
                # Place block
                new_current = current.copy()
                new_current[start:start + block_size] = 1
                
                # Recurse for next block
                next_pos = start + block_size + 1  # +1 for mandatory gap
                place_blocks(next_pos, block_idx + 1, new_current)
        
        place_blocks(0, 0, np.zeros(length, dtype=int))
        return solutions
    
    @staticmethod
    def _is_line_compatible(partial_line: np.ndarray, solution: np.ndarray) -> bool:
        """Check if solution is compatible with partial line"""
        for i in range(len(partial_line)):
            if partial_line[i] != -1 and partial_line[i] != solution[i]:
                return False
        return True
    
    @staticmethod
    def estimate_difficulty(row_constraints: List[List[int]], 
                          col_constraints: List[List[int]]) -> str:
        """Estimate puzzle difficulty based on constraints"""
        height, width = len(row_constraints), len(col_constraints)
        total_cells = height * width
        
        # Calculate constraint complexity
        total_blocks = sum(len(c) for c in row_constraints + col_constraints)
        avg_blocks_per_line = total_blocks / (height + width)
        
        # Calculate density
        total_filled = sum(sum(c) for c in row_constraints + col_constraints) / 2  # Average of row/col totals
        density = total_filled / total_cells
        
        # Size factor
        size_factor = min(height, width)
        
        # Difficulty scoring
        complexity_score = avg_blocks_per_line * 2 + abs(density - 0.5) * 3 + size_factor * 0.5
        
        if complexity_score < 8:
            return "easy"
        elif complexity_score < 15:
            return "medium"
        elif complexity_score < 25:
            return "hard"
        else:
            return "expert"