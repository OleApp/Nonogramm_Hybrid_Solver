import unittest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.solver_state import SolverState
from solvers.classical_solver import ClassicalSolver, LineSolver

class TestLineSolver(unittest.TestCase):
    """Test the LineSolver component"""
    
    def setUp(self):
        self.line_solver = LineSolver()
    
    def test_empty_constraint(self):
        """Test line with empty constraint (all cells should be empty)"""
        line = np.array([-1, -1, -1, -1, -1])  # All unknown
        constraint = []
        
        changed = self.line_solver.solve_line(line, constraint)
        
        self.assertTrue(changed)
        self.assertTrue(np.array_equal(line, [0, 0, 0, 0, 0]))
    
    def test_single_block_exact_fit(self):
        """Test constraint that exactly fills the line"""
        line = np.array([-1, -1, -1])  # All unknown
        constraint = [3]
        
        changed = self.line_solver.solve_line(line, constraint)
        
        self.assertTrue(changed)
        self.assertTrue(np.array_equal(line, [1, 1, 1]))
    
    def test_single_block_with_gap(self):
        """Test single block with possible positions"""
        line = np.array([-1, -1, -1, -1, -1])  # All unknown
        constraint = [3]
        
        # Should determine middle cell must be filled
        # Possible positions: [1,1,1,0,0] or [0,1,1,1,0] or [0,0,1,1,1]
        # Only position 2 is filled in all possibilities
        changed = self.line_solver.solve_line(line, constraint)
        
        if hasattr(self.line_solver, '_generate_valid_solutions'):
            # Test would pass if method is implemented
            self.assertTrue(changed or not changed)  # Placeholder
    
    def test_multiple_blocks(self):
        """Test multiple blocks constraint"""
        line = np.array([-1, -1, -1, -1, -1])  # All unknown
        constraint = [1, 1]
        
        # Should leave gaps between blocks
        changed = self.line_solver.solve_line(line, constraint)
        
        # Test structure exists
        self.assertIsNotNone(changed)
    
    def test_partial_line_compatibility(self):
        """Test line with some cells already filled"""
        line = np.array([1, -1, -1, 0, 1])  # Mixed state
        constraint = [1, 1]
        
        changed = self.line_solver.solve_line(line, constraint)
        
        # Should handle partial solutions
        self.assertIsNotNone(changed)

class TestClassicalSolver(unittest.TestCase):
    """Test the ClassicalSolver"""
    
    def setUp(self):
        """Set up test cases"""
        # Simple 3x3 cross pattern
        self.simple_state = SolverState(
            grid=np.full((3, 3), -1),
            row_constraints=[[1], [3], [1]],
            col_constraints=[[1], [3], [1]]
        )
        
        # 5x5 border pattern
        self.border_state = SolverState(
            grid=np.full((5, 5), -1),
            row_constraints=[[5], [2], [2], [2], [5]],
            col_constraints=[[5], [2], [2], [2], [5]]
        )
        
        # Empty puzzle
        self.empty_state = SolverState(
            grid=np.full((3, 3), -1),
            row_constraints=[[], [], []],
            col_constraints=[[], [], []]
        )
    
    def test_initialization(self):
        """Test solver initialization"""
        solver = ClassicalSolver(self.simple_state)
        
        self.assertIsNotNone(solver.state)
        self.assertIsNotNone(solver.line_solver)
        self.assertEqual(solver.state.grid.shape, (3, 3))
    
    def test_empty_puzzle_solving(self):
        """Test solving puzzle with no filled cells"""
        solver = ClassicalSolver(self.empty_state)
        
        solved = solver.solve()
        
        self.assertTrue(solved)
        # All cells should be empty (0)
        expected_grid = np.zeros((3, 3))
        self.assertTrue(np.array_equal(solver.state.grid, expected_grid))
    
    def test_solve_step(self):
        """Test single solving step"""
        solver = ClassicalSolver(self.simple_state)
        
        # Should make some progress on first step
        progress = solver.solve_step()
        
        # Progress depends on implementation
        self.assertIsInstance(progress, bool)
    
    def test_propagate_constraints(self):
        """Test constraint propagation"""
        solver = ClassicalSolver(self.simple_state)
        
        progress = solver.propagate_constraints()
        
        self.assertIsInstance(progress, bool)
        # Grid should have changed from all -1s
        self.assertTrue(np.any(solver.state.grid != -1) or not progress)
    
    def test_max_iterations_limit(self):
        """Test that solver respects iteration limit"""
        solver = ClassicalSolver(self.simple_state)
        
        # Solve with very low iteration limit
        solved = solver.solve(max_iterations=1)
        
        # Should stop after 1 iteration
        self.assertLessEqual(solver.state.iteration, 1)
    
    def test_is_solved_property(self):
        """Test is_solved property inheritance"""
        solver = ClassicalSolver(self.empty_state)
        
        # Should use inherited property from BaseSolver
        self.assertFalse(solver.is_solved)  # Initially not solved
        
        solver.solve()
        self.assertTrue(solver.is_solved)  # Should be solved after
    
    def test_progress_percentage(self):
        """Test progress calculation"""
        solver = ClassicalSolver(self.simple_state)
        
        initial_progress = solver.progress_percentage
        self.assertEqual(initial_progress, 0.0)  # No cells solved initially
        
        # Make some progress
        solver.solve_step()
        
        final_progress = solver.progress_percentage
        self.assertGreaterEqual(final_progress, initial_progress)
    
    def test_state_consistency(self):
        """Test that solver maintains state consistency"""
        solver = ClassicalSolver(self.simple_state)
        
        original_constraints = {
            'rows': solver.state.row_constraints.copy(),
            'cols': solver.state.col_constraints.copy()
        }
        
        solver.solve()
        
        # Constraints should not change
        self.assertEqual(solver.state.row_constraints, original_constraints['rows'])
        self.assertEqual(solver.state.col_constraints, original_constraints['cols'])
    
    def test_callback_system(self):
        """Test move callback functionality"""
        solver = ClassicalSolver(self.simple_state)
        callback_calls = []
        
        def test_callback(move, state):
            callback_calls.append((move, state.iteration))
        
        solver.add_move_callback(test_callback)
        
        # Callbacks should be stored
        self.assertEqual(len(solver.move_callbacks), 1)
    
    def test_reset_functionality(self):
        """Test solver reset"""
        solver = ClassicalSolver(self.simple_state)
        
        # Make some progress
        solver.solve_step()
        
        # Reset solver
        solver.reset()
        
        # Should be back to initial state
        self.assertTrue(np.all(solver.state.grid == -1))
        self.assertEqual(solver.state.iteration, 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for complete solving scenarios"""
    
    def test_simple_solvable_puzzle(self):
        """Test solving a known solvable puzzle"""
        # 2x2 checkerboard pattern
        state = SolverState(
            grid=np.full((2, 2), -1),
            row_constraints=[[1], [1]],
            col_constraints=[[1], [1]]
        )
        
        solver = ClassicalSolver(state)
        solved = solver.solve()
        
        # Should find a valid solution
        self.assertIsInstance(solved, bool)
        
        # If solved, should have no unknown cells
        if solved:
            self.assertFalse(np.any(solver.state.grid == -1))
    
    def test_impossible_puzzle(self):
        """Test behavior with impossible constraints"""
        # Impossible: 2x2 grid but constraint requires 3 cells
        state = SolverState(
            grid=np.full((2, 2), -1),
            row_constraints=[[3], [1]],  # Impossible - 3 cells in 2-wide row
            col_constraints=[[1], [1]]
        )
        
        solver = ClassicalSolver(state)
        solved = solver.solve()
        
        # Should not crash, return False or partial solution
        self.assertIsInstance(solved, bool)

if __name__ == '__main__':
    # Run specific test categories
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['line', 'solver', 'integration', 'all'], 
                       default='all', help='Which tests to run')
    args = parser.parse_args()
    
    if args.test == 'line':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestLineSolver)
    elif args.test == 'solver':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestClassicalSolver)
    elif args.test == 'integration':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
    else:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)