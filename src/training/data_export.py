import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import pickle

class DataExporter:
    """Export training data in various formats for ML frameworks"""
    
    def __init__(self, training_data: List[Dict[str, Any]]):
        self.training_data = training_data
    
    def export_to_csv(self, output_file: str):
        """Export flattened data to CSV for analysis"""
        rows = []
        
        for session in self.training_data:
            base_row = {
                'puzzle_id': session['puzzle_id'],
                'solved': session['solved'],
                'total_iterations': session['total_iterations'],
                'solver_type': session['solver_type'],
                'grid_height': len(session['constraints']['rows']),
                'grid_width': len(session['constraints']['cols']),
                'total_moves': len(session['moves']),
                'progress_percentage': session.get('stats', {}).get('progress_percentage', 0)
            }
            
            # Add constraint complexity metrics
            row_complexity = self._calculate_constraint_complexity(session['constraints']['rows'])
            col_complexity = self._calculate_constraint_complexity(session['constraints']['cols'])
            
            base_row.update({
                'row_constraint_complexity': row_complexity,
                'col_constraint_complexity': col_complexity,
                'total_constraint_complexity': row_complexity + col_complexity
            })
            
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    
    def export_for_tensorflow(self, output_dir: str):
        """Export data in TensorFlow-compatible format"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Prepare training examples
        examples = []
        labels = []
        
        for session in self.training_data:
            if not session.get('states'):
                continue
                
            for i, state in enumerate(session['states'][:-1]):  # Skip final state
                # Input: current grid state + constraints
                grid = np.array(state['grid'])
                constraints = self._encode_constraints(session['constraints'])
                
                # Output: next state changes
                next_state = np.array(session['states'][i + 1]['grid'])
                changes = (next_state != grid).astype(int)
                
                examples.append({
                    'grid': grid,
                    'constraints': constraints,
                    'certainty': state.get('certainty_map', np.zeros_like(grid))
                })
                labels.append(changes)
        
        # Save as numpy arrays
        np.save(output_path / 'examples.npy', examples)
        np.save(output_path / 'labels.npy', labels)
        
        # Save metadata
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump({
                'num_examples': len(examples),
                'input_shape': examples[0]['grid'].shape if examples else None,
                'data_version': '1.0'
            }, f)
    
    def export_for_pytorch(self, output_file: str):
        """Export data in PyTorch-compatible format"""
        dataset = {
            'puzzles': [],
            'solutions': [],
            'constraints': [],
            'solving_sequences': []
        }
        
        for session in self.training_data:
            if not session['solved']:
                continue  # Only export solved puzzles
                
            dataset['puzzles'].append(np.array(session['initial_state']))
            dataset['solutions'].append(np.array(session['final_state']))
            dataset['constraints'].append(session['constraints'])
            
            # Extract solving sequence
            if session.get('states'):
                sequence = [np.array(state['grid']) for state in session['states']]
                dataset['solving_sequences'].append(sequence)
        
        # Save as pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
    
    def export_puzzle_dataset(self, output_file: str, format: str = 'json'):
        """Export just puzzles and solutions for dataset creation"""
        puzzles = []
        
        for session in self.training_data:
            puzzle_data = {
                'id': session['puzzle_id'],
                'constraints': session['constraints'],
                'solution': session['final_state'] if session['solved'] else None,
                'difficulty': self._estimate_difficulty(session),
                'size': {
                    'height': len(session['constraints']['rows']),
                    'width': len(session['constraints']['cols'])
                }
            }
            puzzles.append(puzzle_data)
        
        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump({'puzzles': puzzles}, f, indent=2)
        elif format == 'csv':
            # Flatten for CSV export
            flat_puzzles = []
            for p in puzzles:
                flat_p = {
                    'id': p['id'],
                    'height': p['size']['height'],
                    'width': p['size']['width'],
                    'difficulty': p['difficulty'],
                    'solved': p['solution'] is not None,
                    'row_constraints': json.dumps(p['constraints']['rows']),
                    'col_constraints': json.dumps(p['constraints']['cols'])
                }
                flat_puzzles.append(flat_p)
            
            df = pd.DataFrame(flat_puzzles)
            df.to_csv(output_file, index=False)
    
    def export_move_sequences(self, output_file: str):
        """Export move sequences for sequence learning"""
        sequences = []
        
        for session in self.training_data:
            if not session.get('moves'):
                continue
                
            sequence = {
                'puzzle_id': session['puzzle_id'],
                'moves': session['moves'],
                'success': session['solved'],
                'sequence_length': len(session['moves'])
            }
            sequences.append(sequence)
        
        with open(output_file, 'w') as f:
            json.dump({'sequences': sequences}, f, indent=2)
    
    def _calculate_constraint_complexity(self, constraints: List[List[int]]) -> float:
        """Calculate complexity metric for constraints"""
        if not constraints:
            return 0.0
            
        total_complexity = 0
        for constraint in constraints:
            # Complexity based on number of blocks and gaps
            if constraint:
                blocks = len(constraint)
                total_filled = sum(constraint)
                # More blocks or higher fill ratio = more complex
                complexity = blocks * 0.5 + (total_filled / 10) * 0.5
                total_complexity += complexity
        
        return total_complexity / len(constraints)
    
    def _encode_constraints(self, constraints: Dict[str, List[List[int]]]) -> np.ndarray:
        """Encode constraints as feature vectors"""
        # Simple encoding - could be made more sophisticated
        row_features = [len(c) for c in constraints['rows']]
        col_features = [len(c) for c in constraints['cols']]
        
        # Pad to same length
        max_len = max(len(row_features), len(col_features))
        row_features.extend([0] * (max_len - len(row_features)))
        col_features.extend([0] * (max_len - len(col_features)))
        
        return np.array(row_features + col_features)
    
    def _estimate_difficulty(self, session: Dict[str, Any]) -> str:
        """Estimate puzzle difficulty based on solving data"""
        if not session['solved']:
            return 'unsolved'
        
        iterations = session['total_iterations']
        moves = len(session.get('moves', []))
        
        if iterations <= 5 and moves <= 10:
            return 'easy'
        elif iterations <= 15 and moves <= 50:
            return 'medium'
        elif iterations <= 50 and moves <= 200:
            return 'hard'
        else:
            return 'expert'