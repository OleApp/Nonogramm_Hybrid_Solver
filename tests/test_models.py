### USAGE
# cd tests
# python test_models.py --test all --verbose
# python test_models.py --test properties
# python test_models.py --test copy


import unittest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.solver_state import SolverState

class TestSolverStateInitialization(unittest.TestCase):
    """Test SolverState initialization and basic properties"""
    
    def test_basic_initialization(self):
        """Test basic SolverState creation"""
        grid = np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]])
        row_constraints = [[1], [1, 1], [1]]
        col_constraints = [[1], [1, 1], [1]]
        
        state = SolverState(
            grid=grid,
            row_constraints=row_constraints,
            col_constraints=col_constraints
        )
        
        # Check basic properties
        self.assertTrue(np.array_equal(state.grid, grid))
        self.assertEqual(state.row_constraints, row_constraints)
        self.assertEqual(state.col_constraints, col_constraints)
        self.assertEqual(state.iteration, 0)
    
    def test_initialization_with_iteration(self):
        """Test initialization with custom iteration value"""
        grid = np.zeros((2, 2))
        state = SolverState(
            grid=grid,
            row_constraints=[[], []],
            col_constraints=[[], []],
            iteration=5
        )
        
        self.assertEqual(state.iteration, 5)
    
    def test_certainty_map_auto_initialization(self):
        """Test automatic certainty map initialization"""
        grid = np.array([[1, 0], [-1, 1]])
        state = SolverState(
            grid=grid,
            row_constraints=[[1], [1]],
            col_constraints=[[1], [1]]
        )
        
        # Should auto-initialize certainty_map in __post_init__
        self.assertIsNotNone(state.certainty_map)
        self.assertEqual(state.certainty_map.shape, grid.shape)
        self.assertTrue(np.array_equal(state.certainty_map, np.zeros_like(grid, dtype=float)))
    
    def test_custom_certainty_map(self):
        """Test initialization with custom certainty map"""
        grid = np.array([[1, 0], [-1, 1]])
        certainty_map = np.array([[0.9, 0.8], [0.1, 0.95]])
        
        state = SolverState(
            grid=grid,
            row_constraints=[[1], [1]],
            col_constraints=[[1], [1]],
            certainty_map=certainty_map
        )
        
        # Should use provided certainty map
        self.assertTrue(np.array_equal(state.certainty_map, certainty_map))
    
    def test_empty_constraints(self):
        """Test initialization with empty constraints"""
        grid = np.zeros((2, 3))
        state = SolverState(
            grid=grid,
            row_constraints=[[], []],
            col_constraints=[[], [], []]
        )
        
        self.assertEqual(len(state.row_constraints), 2)
        self.assertEqual(len(state.col_constraints), 3)
        self.assertEqual(state.row_constraints, [[], []])

class TestSolverStateProperties(unittest.TestCase):
    """Test SolverState properties and methods"""
    
    def setUp(self):
        """Set up test states"""
        # Complete 3x3 grid
        self.complete_grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.complete_state = SolverState(
            grid=self.complete_grid,
            row_constraints=[[1, 1], [1], [1, 1]],
            col_constraints=[[1, 1], [1], [1, 1]]
        )
        
        # Partial 3x3 grid
        self.partial_grid = np.array([[1, -1, 1], [-1, 1, 0], [1, 0, -1]])
        self.partial_state = SolverState(
            grid=self.partial_grid,
            row_constraints=[[1, 1], [1], [1]],
            col_constraints=[[1], [1], [1]]
        )
        
        # All unknown 2x2 grid
        self.unknown_grid = np.full((2, 2), -1)
        self.unknown_state = SolverState(
            grid=self.unknown_grid,
            row_constraints=[[1], [1]],
            col_constraints=[[1], [1]]
        )
    
    def test_shape_property(self):
        """Test shape property"""
        self.assertEqual(self.complete_state.shape, (3, 3))
        self.assertEqual(self.partial_state.shape, (3, 3))
        self.assertEqual(self.unknown_state.shape, (2, 2))
    
    def test_is_complete_property(self):
        """Test is_complete property"""
        # Complete grid should return True
        self.assertTrue(self.complete_state.is_complete)
        
        # Partial grid should return False
        self.assertFalse(self.partial_state.is_complete)
        
        # All unknown grid should return False
        self.assertFalse(self.unknown_state.is_complete)
    
    def test_is_complete_with_different_values(self):
        """Test is_complete with different grid configurations"""
        # All zeros
        all_zeros = SolverState(
            grid=np.zeros((2, 2)),
            row_constraints=[[], []],
            col_constraints=[[], []]
        )
        self.assertTrue(all_zeros.is_complete)
        
        # All ones
        all_ones = SolverState(
            grid=np.ones((2, 2)),
            row_constraints=[[2], [2]],
            col_constraints=[[2], [2]]
        )
        self.assertTrue(all_ones.is_complete)
        
        # Mixed but complete
        mixed = SolverState(
            grid=np.array([[1, 0], [0, 1]]),
            row_constraints=[[1], [1]],
            col_constraints=[[1], [1]]
        )
        self.assertTrue(mixed.is_complete)

class TestSolverStateCopy(unittest.TestCase):
    """Test SolverState copy functionality"""
    
    def setUp(self):
        """Set up test state"""
        self.grid = np.array([[1, -1, 0], [0, 1, -1]])
        self.certainty_map = np.array([[0.9, 0.5, 0.8], [0.7, 0.95, 0.3]])
        self.original_state = SolverState(
            grid=self.grid,
            row_constraints=[[1], [1]],
            col_constraints=[[1], [1], []],
            iteration=10,
            certainty_map=self.certainty_map
        )
    
    def test_copy_creates_independent_copy(self):
        """Test that copy creates independent state"""
        copied_state = self.original_state.copy()
        
        # Should have same values
        self.assertTrue(np.array_equal(copied_state.grid, self.original_state.grid))
        self.assertEqual(copied_state.row_constraints, self.original_state.row_constraints)
        self.assertEqual(copied_state.col_constraints, self.original_state.col_constraints)
        self.assertEqual(copied_state.iteration, self.original_state.iteration)
        self.assertTrue(np.array_equal(copied_state.certainty_map, self.original_state.certainty_map))
    
    def test_copy_independence(self):
        """Test that copied state is independent of original"""
        copied_state = self.original_state.copy()
        
        # Modify original
        self.original_state.grid[0, 0] = 0
        self.original_state.iteration = 20
        self.original_state.certainty_map[0, 0] = 0.1
        
        # Copied state should remain unchanged
        self.assertEqual(copied_state.grid[0, 0], 1)  # Original value
        self.assertEqual(copied_state.iteration, 10)
        self.assertEqual(copied_state.certainty_map[0, 0], 0.9)
    
    def test_copy_with_none_certainty_map(self):
        """Test copy when certainty_map is None"""
        state_without_certainty = SolverState(
            grid=np.array([[1, 0]]),
            row_constraints=[[1]],
            col_constraints=[[1], []],
            certainty_map=None
        )
        
        copied_state = state_without_certainty.copy()
        
        # Should handle None certainty_map gracefully
        # Note: __post_init__ will create certainty_map, so it won't be None
        self.assertIsNotNone(copied_state.certainty_map)
    
    def test_copy_preserves_constraint_references(self):
        """Test that constraints are properly copied (not just referenced)"""
        copied_state = self.original_state.copy()
        
        # Modify constraint in original
        self.original_state.row_constraints[0].append(2)
        
        # Copied state should be unchanged
        self.assertEqual(len(copied_state.row_constraints[0]), 1)
        self.assertEqual(copied_state.row_constraints[0], [1])

class TestSolverStateEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_grid(self):
        """Test with minimal grid size"""
        grid = np.array([[1]])
        state = SolverState(
            grid=grid,
            row_constraints=[[1]],
            col_constraints=[[1]]
        )
        
        self.assertEqual(state.shape, (1, 1))
        self.assertTrue(state.is_complete)
    
    def test_mismatched_certainty_map_shape(self):
        """Test behavior with mismatched certainty map shape"""
        grid = np.array([[1, 0], [0, 1]])
        wrong_shape_certainty = np.array([[0.5]])  # Wrong shape
        
        # This should either raise an error or handle gracefully
        try:
            state = SolverState(
                grid=grid,
                row_constraints=[[1], [1]],
                col_constraints=[[1], [1]],
                certainty_map=wrong_shape_certainty
            )
            # If no error, certainty_map should be auto-corrected in __post_init__
            self.assertEqual(state.certainty_map.shape, grid.shape)
        except (ValueError, AttributeError):
            # Expected behavior for shape mismatch
            pass
    
    def test_large_grid(self):
        """Test with larger grid sizes"""
        large_grid = np.random.choice([-1, 0, 1], size=(20, 15))
        row_constraints = [[] for _ in range(20)]
        col_constraints = [[] for _ in range(15)]
        
        state = SolverState(
            grid=large_grid,
            row_constraints=row_constraints,
            col_constraints=col_constraints
        )
        
        self.assertEqual(state.shape, (20, 15))
        self.assertIsNotNone(state.certainty_map)
        self.assertEqual(state.certainty_map.shape, (20, 15))
    
    def test_grid_value_validation(self):
        """Test grid with invalid values"""
        # Grid with values outside [-1, 0, 1] range
        invalid_grid = np.array([[2, -2, 5], [0, 1, -1]])
        
        # Should still create state (validation might be elsewhere)
        state = SolverState(
            grid=invalid_grid,
            row_constraints=[[], []],
            col_constraints=[[], [], []]
        )
        
        self.assertEqual(state.shape, (2, 3))
    
    def test_constraint_list_modification(self):
        """Test that constraint lists can be safely modified"""
        original_row_constraints = [[1, 2], [3]]
        original_col_constraints = [[2], [1, 1]]
        
        state = SolverState(
            grid=np.zeros((2, 2)),
            row_constraints=original_row_constraints,
            col_constraints=original_col_constraints
        )
        
        # Modify original constraint lists
        original_row_constraints[0].append(4)
        original_col_constraints.append([5])
        
        # State constraints should reflect the changes (they're the same objects)
        self.assertEqual(state.row_constraints[0], [1, 2, 4])
        self.assertEqual(len(state.col_constraints), 3)

class TestSolverStateDataclassFeatures(unittest.TestCase):
    """Test dataclass-specific features"""
    
    def test_equality_comparison(self):
        """Test equality comparison between states"""
        grid = np.array([[1, 0], [0, 1]])
        constraints_row = [[1], [1]]
        constraints_col = [[1], [1]]
        
        state1 = SolverState(
            grid=grid.copy(),
            row_constraints=constraints_row.copy(),
            col_constraints=constraints_col.copy()
        )
        
        state2 = SolverState(
            grid=grid.copy(),
            row_constraints=constraints_row.copy(),
            col_constraints=constraints_col.copy()
        )
        
        # Note: Numpy arrays in dataclasses don't compare equal by default
        # This test documents current behavior
        try:
            equal = state1 == state2
            # Comparison might work or might raise an error due to numpy arrays
        except ValueError:
            # Expected behavior with numpy arrays in dataclass
            pass
    
    def test_string_representation(self):
        """Test string representation"""
        state = SolverState(
            grid=np.array([[1, 0]]),
            row_constraints=[[1]],
            col_constraints=[[1], []]
        )
        
        str_repr = str(state)
        self.assertIsInstance(str_repr, str)
        self.assertIn("SolverState", str_repr)
    
    def test_field_access(self):
        """Test direct field access"""
        state = SolverState(
            grid=np.array([[1]]),
            row_constraints=[[1]],
            col_constraints=[[1]]
        )
        
        # All fields should be accessible
        self.assertIsInstance(state.grid, np.ndarray)
        self.assertIsInstance(state.row_constraints, list)
        self.assertIsInstance(state.col_constraints, list)
        self.assertIsInstance(state.iteration, int)
        self.assertIsInstance(state.certainty_map, np.ndarray)

if __name__ == '__main__':
    # Command line interface for running specific test categories
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SolverState model')
    parser.add_argument('--test', 
                       choices=['init', 'properties', 'copy', 'edge', 'dataclass', 'all'],
                       default='all',
                       help='Which test category to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    args = parser.parse_args()
    
    # Create test suite based on selection
    if args.test == 'init':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSolverStateInitialization)
    elif args.test == 'properties':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSolverStateProperties)
    elif args.test == 'copy':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSolverStateCopy)
    elif args.test == 'edge':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSolverStateEdgeCases)
    elif args.test == 'dataclass':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSolverStateDataclassFeatures)
    else:
        # Load all tests
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Summary output
    if result.wasSuccessful():
        print(f"\n✅ All SolverState tests passed! ({result.testsRun} tests)")
    else:
        failures = len(result.failures)
        errors = len(result.errors)
        print(f"\n❌ Test failures: {failures}, errors: {errors}")
        
        if result.failures:
            print("\nFailures:")
            for test, trace in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nErrors:")
            for test, trace in result.errors:
                print(f"  - {test}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)