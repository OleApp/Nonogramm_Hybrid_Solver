import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from models.solver_state import SolverState

class TrainingDataCollector:
    def __init__(self):
        self.training_data = []
        self.current_session = None
    
    def start_session(self, solver, puzzle_id: Optional[str] = None) -> Dict[str, Any]:
        """Start collecting data for a solving session"""
        session_data = {
            'puzzle_id': puzzle_id or f"puzzle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'initial_state': solver.state.grid.tolist(),
            'constraints': {
                'rows': solver.state.row_constraints,
                'cols': solver.state.col_constraints
            },
            'moves': [],
            'states': [],
            'final_state': None,
            'solved': False,
            'solver_type': type(solver).__name__,
            'total_iterations': 0
        }
        
        # Hook into solver to collect moves
        solver.add_move_callback(self._collect_move)
        self.current_session = session_data
        
        return session_data
    
    def _collect_move(self, move: Any, state: SolverState):
        """Callback to collect individual moves"""
        if self.current_session is None:
            return
            
        self.current_session['moves'].append({
            'move_type': type(move).__name__ if hasattr(move, '__class__') else str(move),
            'move_data': move if isinstance(move, (dict, list, str, int)) else str(move),
            'iteration': state.iteration
        })
        
        # Store grid state snapshot
        self.current_session['states'].append({
            'grid': state.grid.tolist(),
            'certainty_map': state.certainty_map.tolist() if state.certainty_map is not None else None,
            'iteration': state.iteration
        })
    
    def end_session(self, solver) -> bool:
        """End the current session and store results"""
        if self.current_session is None:
            return False
        
        self.current_session['solved'] = solver.is_solved
        self.current_session['final_state'] = solver.state.grid.tolist()
        self.current_session['total_iterations'] = solver.state.iteration
        
        # Calculate solving statistics
        self.current_session['stats'] = self._calculate_stats(solver)
        
        self.training_data.append(self.current_session)
        solved = self.current_session['solved']
        self.current_session = None
        
        return solved
    
    def collect_solve_session(self, solver, puzzle_id: Optional[str] = None) -> bool:
        """Complete wrapper: start session, solve, end session"""
        self.start_session(solver, puzzle_id)
        solved = solver.solve()
        self.end_session(solver)
        return solved
    
    def _calculate_stats(self, solver) -> Dict[str, Any]:
        """Calculate solving statistics"""
        grid = solver.state.grid
        total_cells = grid.size
        solved_cells = np.sum(grid != -1)
        
        return {
            'progress_percentage': (solved_cells / total_cells) * 100,
            'cells_solved': int(solved_cells),
            'total_cells': int(total_cells),
            'moves_count': len(self.current_session['moves']),
            'efficiency': solved_cells / max(1, len(self.current_session['moves']))
        }
    
    def save_training_data(self, filename: str):
        """Save collected training data to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                'metadata': {
                    'total_sessions': len(self.training_data),
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0'
                },
                'sessions': self.training_data
            }, f, indent=2)
    
    def load_training_data(self, filename: str):
        """Load training data from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
            self.training_data = data.get('sessions', [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about collected data"""
        if not self.training_data:
            return {}
        
        solved_count = sum(1 for session in self.training_data if session['solved'])
        total_moves = sum(len(session['moves']) for session in self.training_data)
        
        return {
            'total_sessions': len(self.training_data),
            'solved_sessions': solved_count,
            'success_rate': (solved_count / len(self.training_data)) * 100,
            'average_moves': total_moves / len(self.training_data),
            'total_moves': total_moves
        }
    
    def clear_data(self):
        """Clear all collected training data"""
        self.training_data = []
        self.current_session = None