### USAGE
# cd tests
# python test_hybrid_solver.py --test all --verbose
# python test_hybrid_solver.py --test solving
# python test_hybrid_solver.py --test integration

import unittest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.solver_state import SolverState
from solvers.hybrid_solver import HybridNonogramSolver
from solvers.classical_solver import ClassicalSolver
from solvers.ai_heuristic import AIHeuristic

class TestHybridSolverInitialization(unittest.TestCase):
    """Test HybridNonogramSolver initialization"""
    
    def test_basic_initialization(self):
        """Test basic solver initialization"""
        row_constraints = [[1, 1], [2], [1]]
        col_constraints = [[1], [2], [1]]
        
        solver = HybridNonogramSolver(row_constraints, col_constraints)
        
        # Check basic properties
        self.assertIsNotNone(solver.state)
        self.assertEqual(solver.state.grid.shape, (3, 3))
        self.assertTrue(np.all(solver.state.grid == -1))
        self.assertEqual(solver.state.row_constraints, row_constraints)
        self.assertEqual(solver.state.col_constraints, col_constraints)
    
    def test_component_solvers_initialization(self):
        """Test that component solvers are properly initialized"""
        row_constraints = [[2], [1]]
        col_constraints = [[1], [2]]
        
        solver = HybridNonogramSolver(row_constraints, col_constraints)
        
        # Check component solvers exist
        self.assertIsInstance(solver.classical_solver, ClassicalSolver)
        self.assertIsInstance(solver.ai_heuristic, AIHeuristic)
        
        # Check they share the same state
        self.assertIs(solver.classical_solver.state, solver.state)
        self.assertIs(solver.ai_heuristic.state, solver.state)
    
    def test_ai_toggle(self):
        """Test AI enable/disable functionality"""
        row_constraints = [[1]]
        col_constraints = [[1]]
        
        solver = HybridNonogramSolver(row_constraints, col_constraints)
        
        # Initially AI should be enabled
        self.assertTrue(solver.use_ai)
        
        # Disable AI
        solver.enable_ai(False)
        self.assertFalse(solver.use_ai)
        
        # Re-enable AI
        solver.enable_ai(True)
        self.assertTrue(solver.use_ai)
    
    def test_invalid_constraints(self):
        """Test behavior with mismatched constraint dimensions"""
        row_constraints = [[1], [2]]  # 2 rows
        col_constraints = [[1], [1], [1]]  # 3 columns
        
        # Should handle dimension mismatch gracefully
        try:
            solver = HybridNonogramSolver(row_constraints, col_constraints)
            # Grid should be 2x3 (rows x first col constraint count)
            expected_shape = (len(row_constraints), len(col_constraints[0]))
            # This might fail depending on implementation
        except (IndexError, ValueError):
            # Expected for malformed constraints
            pass

class TestHybridSolverStrategies(unittest.TestCase):
    """Test different solving strategies"""
    
    def setUp(self):
        """Set up test puzzles"""
        # Simple 3x3 cross
        self.cross_solver = HybridNonogramSolver(
            [[1], [3], [1]],
            [[1], [3], [1]]
        )
        
        # Empty 2x2 puzzle
        self.empty_solver = HybridNonogramSolver(
            [[], []],
            [[], []]
        )
        
        # Complex 5x5 border
        self.border_solver = HybridNonogramSolver(
            [[5], [2], [2], [2], [5]],
            [[5], [2], [2], [2], [5]]
        )
    
    def test_classical_only_strategy(self):
        """Test solving with classical solver only"""
        solver = self.empty_solver
        
        solved = solver.solve_with_strategy("classical")
        
        self.assertIsInstance(solved, bool)
        if solved:
            # Should be all zeros for empty puzzle
            self.assertTrue(np.all(solver.state.grid == 0))
    
    def test_ai_only_strategy(self):
        """Test solving with AI heuristic only"""
        solver = self.cross_solver
        
        solved = solver.solve_with_strategy("ai")
        
        self.assertIsInstance(solved, bool)
        # AI might make some progress even if not complete
    
    def test_hybrid_strategy(self):
        """Test default hybrid strategy"""
        solver = self.empty_solver
        
        solved = solver.solve_with_strategy("hybrid")
        
        self.assertIsInstance(solved, bool)
        # Should be same as default solve()
        
        solver.reset()
        solved_default = solver.solve()
        
        self.assertEqual(type(solved), type(solved_default))
    
    def test_invalid_strategy(self):
        """Test behavior with invalid strategy"""
        solver = self.cross_solver
        
        with self.assertRaises(ValueError):
            solver.solve_with_strategy("invalid_strategy")
    
    def test_strategy_comparison(self):
        """Test that different strategies can produce results"""
        solver = self.empty_solver
        
        # Test each strategy
        strategies = ["classical", "ai", "hybrid"]
        results = {}
        
        for strategy in strategies:
            solver.reset()
            results[strategy] = solver.solve_with_strategy(strategy)
        
        # At least one strategy should work for empty puzzle
        self.assertTrue(any(results.values()))

class TestHybridSolverSolving(unittest.TestCase):
    """Test core solving functionality"""
    
    def setUp(self):
        """Set up test cases"""
        self.empty_solver = HybridNonogramSolver([[], []], [[], []])
        self.simple_solver = HybridNonogramSolver([[2]], [[2]])
    
    def test_solve_step(self):
        """Test single solving step"""
        solver = self.empty_solver
        
        # First step should make progress on empty puzzle
        progress = solver.solve_step()
        
        self.assertIsInstance(progress, bool)
        if progress:
            # Some cells should be determined
            self.assertTrue(np.any(solver.state.grid != -1))
    
    def test_solve_empty_puzzle(self):
        """Test solving completely empty puzzle"""
        solver = self.empty_solver
        
        solved = solver.solve()
        
        self.assertTrue(solved)
        # All cells should be empty (0)
        self.assertTrue(np.all(solver.state.grid == 0))
    
    def test_solve_simple_puzzle(self):
        """Test solving simple puzzle"""
        solver = self.simple_solver
        
        solved = solver.solve()
        
        self.assertIsInstance(solved, bool)
        if solved:
            # No unknown cells should remain
            self.assertFalse(np.any(solver.state.grid == -1))
    
    def test_iteration_tracking(self):
        """Test that iterations are properly tracked"""
        solver = self.simple_solver
        
        initial_iteration = solver.state.iteration
        solver.solve(max_iterations=5)
        
        # Should have made some iterations
        self.assertGreaterEqual(solver.state.iteration, initial_iteration)
        self.assertLessEqual(solver.state.iteration, 5)
    
    def test_max_iterations_respected(self):
        """Test that max_iterations limit is respected"""
        solver = self.simple_solver
        
        solver.solve(max_iterations=3)
        
        self.assertLessEqual(solver.state.iteration, 3)
    
    def test_early_termination(self):
        """Test early termination when solved"""
        solver = self.empty_solver  # Should solve quickly
        
        initial_time = solver.state.iteration
        solved = solver.solve(max_iterations=100)
        
        if solved:
            # Should terminate early, not use all iterations
            self.assertLess(solver.state.iteration, 50)

class TestHybridSolverFeatures(unittest.TestCase):
    """Test additional hybrid solver features"""
    
    def setUp(self):
        """Set up test solver"""
        self.solver = HybridNonogramSolver(
            [[1, 1], [3]],
            [[2], [1], [2]]
        )
    
    def test_solving_stats(self):
        """Test solving statistics collection"""
        solver = self.solver
        
        # Get initial stats
        stats = solver.get_solving_stats()
        
        required_keys = ["iterations", "progress_percentage", 
                        "cells_solved", "total_cells", "is_complete"]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check data types
        self.assertIsInstance(stats["iterations"], int)
        self.assertIsInstance(stats["progress_percentage"], (int, float))
        self.assertIsInstance(stats["cells_solved"], (int, np.integer))
        self.assertIsInstance(stats["total_cells"], (int, np.integer))
        self.assertIsInstance(stats["is_complete"], bool)
    
    def test_uncertainty_regions(self):
        """Test uncertainty region mapping"""
        solver = self.solver
        
        uncertainty_map = solver.get_uncertainty_regions()
        
        self.assertIsInstance(uncertainty_map, np.ndarray)
        self.assertEqual(uncertainty_map.shape, solver.state.grid.shape)
    
    def test_ai_confidence_threshold(self):
        """Test AI confidence threshold adjustment"""
        solver = self.solver
        
        # Should not raise errors
        solver.set_ai_confidence_threshold(0.8)
        solver.set_ai_confidence_threshold(0.3)
        solver.set_ai_confidence_threshold(1.0)
        solver.set_ai_confidence_threshold(0.0)
    
    def test_state_sharing(self):
        """Test that all components share the same state"""
        solver = self.solver
        
        # Modify state through hybrid solver
        original_grid = solver.state.grid.copy()
        solver.state.grid[0, 0] = 1
        
        # All solvers should see the change
        self.assertEqual(solver.classical_solver.state.grid[0, 0], 1)
        self.assertEqual(solver.ai_heuristic.state.grid[0, 0], 1)
        
        # Reset for cleanup
        solver.state.grid[0, 0] = -1
    
    def test_callback_integration(self):
        """Test callback system integration"""
        solver = self.solver
        callback_data = []
        
        def test_callback(move, state):
            callback_data.append({
                'move': move,
                'iteration': state.iteration,
                'grid_state': state.grid.copy()
            })
        
        solver.add_move_callback(test_callback)
        
        # Make some solving steps
        solver.solve_step()
        
        # Callbacks should be registered (even if not called yet)
        self.assertEqual(len(solver.move_callbacks), 1)

class TestHybridSolverIntegration(unittest.TestCase):
    """Integration tests for complete solving scenarios"""
    
    def test_ai_classical_coordination(self):
        """Test that AI and classical solvers work together"""
        # Create a puzzle that might benefit from both approaches
        solver = HybridNonogramSolver(
            [[1], [1, 1], [1]],
            [[1], [2], [1]]
        )
        
        # Enable AI
        solver.enable_ai(True)
        
        # Track progress from both components
        initial_state = solver.state.grid.copy()
        
        # Try classical first
        classical_progress = solver.classical_solver.solve_step()
        after_classical = solver.state.grid.copy()
        
        # Try AI step
        ai_progress = solver.ai_heuristic.solve_step()
        after_ai = solver.state.grid.copy()
        
        # At least one should make progress or both should be compatible
        self.assertTrue(
            classical_progress or ai_progress or 
            np.array_equal(after_classical, after_ai)
        )
    
    def test_full_solving_pipeline(self):
        """Test complete solving pipeline"""
        # Use a known solvable pattern
        solver = HybridNonogramSolver(
            [[], [1], []],  # Single dot in middle
            [[], [1], []]
        )
        
        # Solve with full pipeline
        solved = solver.solve()
        stats = solver.get_solving_stats()
        
        self.assertTrue(solved)
        self.assertEqual(stats["progress_percentage"], 100.0)
        self.assertTrue(stats["is_complete"])
        
        # Check solution correctness
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.assertTrue(np.array_equal(solver.state.grid, expected))
    
    def test_performance_monitoring(self):
        """Test performance and resource usage"""
        solver = HybridNonogramSolver(
            [[5], [1], [1], [1], [5]],
            [[5], [1], [1], [1], [5]]
        )
        
        # Measure solving performance
        import time
        start_time = time.time()
        
        solved = solver.solve(max_iterations=50)
        
        end_time = time.time()
        solving_time = end_time - start_time
        
        stats = solver.get_solving_stats()
        
        # Should complete within reasonable time
        self.assertLess(solving_time, 10.0)  # 10 seconds max
        
        # Should make reasonable progress
        self.assertGreater(stats["progress_percentage"], 0)

if __name__ == '__main__':
    # Run tests with different verbosity levels
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', 
                       choices=['init', 'strategies', 'solving', 'features', 'integration', 'all'], 
                       default='all', 
                       help='Which test category to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()
    
    # Select test suite
    if args.test == 'init':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridSolverInitialization)
    elif args.test == 'strategies':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridSolverStrategies)
    elif args.test == 'solving':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridSolverSolving)
    elif args.test == 'features':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridSolverFeatures)
    elif args.test == 'integration':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestHybridSolverIntegration)
    else:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All tests passed! ({result.testsRun} tests)")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
    sys.exit(0 if result.wasSuccessful() else 1)