### USAGE
# # Einfaches Puzzle lösen
# python main.py solve --sample medium --show-progress

# # Puzzle aus Datei mit Classical Solver
# python main.py solve --file my_puzzle.json --solver classical

# # Dataset generieren
# python main.py generate --count 200 --difficulty --output dataset.json

# # Trainingsdaten sammeln
# python main.py collect --count 100 --output training_data.json


#!/usr/bin/env python3
"""
Nonogramm Hybrid Solver - Main Entry Point

This script provides a command-line interface for solving nonogram puzzles
using classical algorithms, AI heuristics, or hybrid approaches.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Core imports
from models.solver_state import SolverState
from solvers.hybrid_solver import HybridNonogramSolver
from solvers.classical_solver import ClassicalSolver
from solvers.ai_heuristic import AIHeuristic

# Training and data imports
from training.data_collector import TrainingDataCollector
from training.dataset_generator import DatasetGenerator
from training.data_export import DataExporter

# Utility imports
from utils.validation import NonogramValidator
from visualisation.grid_display import GridDisplayer, quick_display

def load_puzzle_from_file(filename: str) -> dict:
    """Load puzzle from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Handle different JSON formats
        if 'puzzles' in data:
            # Multiple puzzles format
            return data['puzzles'][0]  # Use first puzzle
        elif 'row_constraints' in data and 'col_constraints' in data:
            # Single puzzle format
            return data
        else:
            raise ValueError("Invalid puzzle format")
            
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading puzzle: {e}")
        sys.exit(1)

def create_sample_puzzle(size: str = "small") -> dict:
    """Create a sample puzzle for testing"""
    if size == "small":
        return {
            'row_constraints': [[1], [3], [1]],
            'col_constraints': [[1], [3], [1]],
            'name': 'Small Cross'
        }
    elif size == "medium":
        return {
            'row_constraints': [[5], [2], [2], [2], [5]],
            'col_constraints': [[5], [2], [2], [2], [5]],
            'name': 'Medium Border'
        }
    elif size == "large":
        return {
            'row_constraints': [[1, 1, 1], [1, 1, 1], [3, 3], [1, 1], [1, 1], [3, 3], [1, 1, 1], [1, 1, 1]],
            'col_constraints': [[1, 1, 1], [1, 1, 1], [3, 3], [1, 1], [1, 1], [3, 3], [1, 1, 1], [1, 1, 1]],
            'name': 'Large Pattern'
        }
    else:
        raise ValueError(f"Unknown size: {size}")

def solve_puzzle(args):
    """Main puzzle solving function"""
    # Load or create puzzle
    if args.file:
        puzzle_data = load_puzzle_from_file(args.file)
        puzzle_name = puzzle_data.get('name', Path(args.file).stem)
    else:
        puzzle_data = create_sample_puzzle(args.sample)
        puzzle_name = puzzle_data['name']
    
    print(f"Solving puzzle: {puzzle_name}")
    print(f"Size: {len(puzzle_data['row_constraints'])}x{len(puzzle_data['col_constraints'])}")
    
    # Validate constraints
    grid_size = (len(puzzle_data['row_constraints']), len(puzzle_data['col_constraints']))
    if not NonogramValidator.validate_constraints(
        puzzle_data['row_constraints'], 
        puzzle_data['col_constraints'], 
        grid_size
    ):
        print("❌ Invalid puzzle constraints!")
        return False
    
    # Create solver
    if args.solver == "hybrid":
        solver = HybridNonogramSolver(
            puzzle_data['row_constraints'],
            puzzle_data['col_constraints']
        )
        solver.enable_ai(not args.no_ai)
        if args.ai_threshold:
            solver.set_ai_confidence_threshold(args.ai_threshold)
    else:
        # Create initial state for other solvers
        import numpy as np
        initial_grid = np.full(grid_size, -1)
        state = SolverState(
            grid=initial_grid,
            row_constraints=puzzle_data['row_constraints'],
            col_constraints=puzzle_data['col_constraints']
        )
        
        if args.solver == "classical":
            solver = ClassicalSolver(state)
        elif args.solver == "ai":
            solver = AIHeuristic(state)
        else:
            raise ValueError(f"Unknown solver: {args.solver}")
    
    # Set up data collection if requested
    collector = None
    if args.collect_data:
        collector = TrainingDataCollector()
        collector.start_session(solver, puzzle_name)
    
    # Show initial state if requested
    if args.show_progress:
        displayer = GridDisplayer()
        print("\nInitial state:")
        displayer.display_puzzle(solver.state, f"{puzzle_name} - Initial")
    
    # Solve puzzle
    print(f"\nSolving with {args.solver} solver...")
    start_time = time.time()
    
    try:
        solved = solver.solve(max_iterations=args.max_iterations)
        solve_time = time.time() - start_time
        
        # Get solving statistics
        if hasattr(solver, 'get_solving_stats'):
            stats = solver.get_solving_stats()
        else:
            stats = {
                'iterations': solver.state.iteration,
                'progress_percentage': solver.progress_percentage,
                'is_complete': solver.is_solved
            }
        
        # Display results
        print(f"\n{'✅ SOLVED' if solved else '⚠️  PARTIAL SOLUTION'}")
        print(f"Time: {solve_time:.2f}s")
        print(f"Iterations: {stats['iterations']}")
        print(f"Progress: {stats['progress_percentage']:.1f}%")
        
        # Show final state
        if args.show_progress or args.show_result:
            print("\nFinal state:")
            displayer = GridDisplayer()
            title = f"{puzzle_name} - {'SOLVED' if solved else 'PARTIAL'}"
            displayer.display_puzzle(solver.state, title)
        
        # Show AI certainty map if available
        if args.show_certainty and hasattr(solver, 'state') and solver.state.certainty_map is not None:
            displayer.display_certainty_map(solver.state, f"{puzzle_name} - AI Certainty")
        
        # Save result if requested
        if args.output:
            save_result(solver, puzzle_name, args.output, solved, stats)
        
        # End data collection
        if collector:
            collector.end_session(solver)
            if args.collect_data != True:  # If specific filename given
                collector.save_training_data(args.collect_data)
                print(f"Training data saved to: {args.collect_data}")
        
        return solved
        
    except KeyboardInterrupt:
        print("\n🛑 Solving interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error during solving: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return False

def save_result(solver, puzzle_name: str, output_path: str, solved: bool, stats: dict):
    """Save solving result to file"""
    result_data = {
        'puzzle_name': puzzle_name,
        'solver_type': type(solver).__name__,
        'solved': solved,
        'final_grid': solver.state.grid.tolist(),
        'statistics': stats,
        'constraints': {
            'rows': solver.state.row_constraints,
            'cols': solver.state.col_constraints
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"Result saved to: {output_path}")
    except Exception as e:
        print(f"Error saving result: {e}")

def generate_dataset(args):
    """Generate training dataset"""
    print(f"Generating {args.count} puzzles...")
    
    generator = DatasetGenerator(seed=args.seed)
    
    if args.difficulty:
        puzzles = generator.generate_difficulty_graded_dataset(args.count // 4)
    else:
        size_range = tuple(map(int, args.size_range.split(',')))
        density_range = tuple(map(float, args.density_range.split(',')))
        
        puzzles = generator.generate_dataset(
            num_puzzles=args.count,
            size_range=size_range,
            density_range=density_range
        )
    
    # Validate puzzles if requested
    if args.validate:
        print("Validating generated puzzles...")
        valid_puzzles = []
        for puzzle in puzzles:
            if generator.validate_puzzle(puzzle):
                valid_puzzles.append(puzzle)
        
        print(f"Valid puzzles: {len(valid_puzzles)}/{len(puzzles)}")
        puzzles = valid_puzzles
    
    # Save dataset
    output_file = args.output or f"dataset_{len(puzzles)}_puzzles.json"
    generator.save_dataset(puzzles, output_file)
    
    print(f"Dataset saved to: {output_file}")
    return True

def collect_training_data(args):
    """Collect training data by solving multiple puzzles"""
    # Load puzzles
    if args.dataset:
        with open(args.dataset, 'r') as f:
            data = json.load(f)
            puzzles = data.get('puzzles', [])
    else:
        # Generate puzzles on the fly
        generator = DatasetGenerator(seed=args.seed)
        puzzles = generator.generate_dataset(args.count, (5, 10), (0.3, 0.7))
    
    print(f"Collecting training data from {len(puzzles)} puzzles...")
    
    # Set up collector
    collector = TrainingDataCollector()
    successful_solves = 0
    
    for i, puzzle_data in enumerate(puzzles):
        print(f"Solving puzzle {i+1}/{len(puzzles)}: ", end='', flush=True)
        
        try:
            # Create solver
            solver = HybridNonogramSolver(
                puzzle_data['row_constraints'],
                puzzle_data['col_constraints']
            )
            
            # Collect solving session
            solved = collector.collect_solve_session(solver, f"puzzle_{i:04d}")
            
            if solved:
                successful_solves += 1
                print("✅")
            else:
                print("⚠️")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Save collected data
    output_file = args.output or f"training_data_{len(puzzles)}_sessions.json"
    collector.save_training_data(output_file)
    
    # Show statistics
    stats = collector.get_statistics()
    print(f"\n📊 Collection Statistics:")
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Successful solves: {stats['solved_sessions']} ({stats['success_rate']:.1f}%)")
    print(f"Average moves per session: {stats['average_moves']:.1f}")
    print(f"Training data saved to: {output_file}")
    
    return True

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Nonogramm Hybrid Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve a sample puzzle with hybrid solver
  python main.py solve --sample medium --show-progress
  
  # Solve from file with classical solver only
  python main.py solve --file puzzle.json --solver classical
  
  # Generate training dataset
  python main.py generate --count 100 --difficulty
  
  # Collect training data
  python main.py collect --count 50 --output training.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve a nonogram puzzle')
    solve_parser.add_argument('--file', '-f', help='Puzzle file (JSON format)')
    solve_parser.add_argument('--sample', choices=['small', 'medium', 'large'], 
                             default='small', help='Use sample puzzle')
    solve_parser.add_argument('--solver', choices=['hybrid', 'classical', 'ai'], 
                             default='hybrid', help='Solver type')
    solve_parser.add_argument('--max-iterations', type=int, default=100,
                             help='Maximum solver iterations')
    solve_parser.add_argument('--no-ai', action='store_true',
                             help='Disable AI component in hybrid solver')
    solve_parser.add_argument('--ai-threshold', type=float,
                             help='AI confidence threshold (0.0-1.0)')
    solve_parser.add_argument('--show-progress', action='store_true',
                             help='Show solving progress visualization')
    solve_parser.add_argument('--show-result', action='store_true',
                             help='Show final result visualization')
    solve_parser.add_argument('--show-certainty', action='store_true',
                             help='Show AI certainty map')
    solve_parser.add_argument('--collect-data', nargs='?', const=True,
                             help='Collect training data (optionally specify filename)')
    solve_parser.add_argument('--output', '-o', help='Save result to file')
    solve_parser.add_argument('--debug', action='store_true',
                             help='Show debug information')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate puzzle dataset')
    gen_parser.add_argument('--count', type=int, default=50,
                           help='Number of puzzles to generate')
    gen_parser.add_argument('--size-range', default='5,15',
                           help='Size range (min,max)')
    gen_parser.add_argument('--density-range', default='0.3,0.7',
                           help='Density range (min,max)')
    gen_parser.add_argument('--difficulty', action='store_true',
                           help='Generate difficulty-graded dataset')
    gen_parser.add_argument('--validate', action='store_true',
                           help='Validate generated puzzles')
    gen_parser.add_argument('--output', '-o', help='Output filename')
    gen_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect training data')
    collect_parser.add_argument('--dataset', help='Dataset file to solve')
    collect_parser.add_argument('--count', type=int, default=20,
                               help='Number of puzzles to solve (if no dataset)')
    collect_parser.add_argument('--output', '-o', help='Output filename')
    collect_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'solve':
            success = solve_puzzle(args)
        elif args.command == 'generate':
            success = generate_dataset(args)
        elif args.command == 'collect':
            success = collect_training_data(args)
        else:
            print(f"Unknown command: {args.command}")
            success = False
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.command == 'solve' and args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()