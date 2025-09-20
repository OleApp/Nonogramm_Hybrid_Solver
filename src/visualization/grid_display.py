import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Tuple
from models.solver_state import SolverState

class GridDisplayer:
    """Visualization utilities for nonogram grids"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        self.colors = {
            -1: 'lightgray',  # Unknown
            0: 'white',       # Empty
            1: 'black'        # Filled
        }
    
    def display_puzzle(self, state: SolverState, title: str = "Nonogram Puzzle"):
        """Display the current puzzle state with constraints"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Draw grid
        self._draw_grid(ax, state.grid)
        
        # Add constraints
        self._add_constraints(ax, state.row_constraints, state.col_constraints, state.grid.shape)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
    def display_solution_comparison(self, initial_state: SolverState, 
                                  solved_state: SolverState, 
                                  solution: Optional[np.ndarray] = None):
        """Display initial state, solved state, and optional target solution"""
        num_plots = 3 if solution is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(self.figsize[0] * num_plots / 2, self.figsize[1]))
        
        if num_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Initial state
        self._draw_grid(axes[0], initial_state.grid)
        self._add_constraints(axes[0], initial_state.row_constraints, 
                            initial_state.col_constraints, initial_state.grid.shape)
        axes[0].set_title("Initial State", fontweight='bold')
        
        # Solved state
        self._draw_grid(axes[1], solved_state.grid)
        axes[1].set_title("Solved State", fontweight='bold')
        
        # Target solution (if provided)
        if solution is not None:
            self._draw_grid(axes[2], solution)
            axes[2].set_title("Target Solution", fontweight='bold')
        
        for ax in axes:
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def display_solving_progress(self, states: List[np.ndarray], 
                               constraints: Tuple[List[List[int]], List[List[int]]],
                               max_frames: int = 9):
        """Display solving progress as animation frames"""
        num_states = min(len(states), max_frames)
        cols = 3
        rows = (num_states + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0], self.figsize[1] * rows / 2))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        row_constraints, col_constraints = constraints
        
        for i in range(num_states):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Select state (evenly distributed across solving process)
            state_idx = int(i * (len(states) - 1) / (num_states - 1)) if num_states > 1 else 0
            
            self._draw_grid(ax, states[state_idx])
            if i == 0:  # Add constraints only to first frame
                self._add_constraints(ax, row_constraints, col_constraints, states[state_idx].shape)
            
            ax.set_title(f"Step {state_idx + 1}", fontsize=10)
            ax.set_aspect('equal')
        
        # Hide unused subplots
        for i in range(num_states, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def display_certainty_map(self, state: SolverState, title: str = "AI Certainty Map"):
        """Display AI certainty/confidence map"""
        if state.certainty_map is None:
            print("No certainty map available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] / 2))
        
        # Current grid state
        self._draw_grid(ax1, state.grid)
        ax1.set_title("Current State")
        
        # Certainty heatmap
        im = ax2.imshow(state.certainty_map, cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_title("AI Certainty")
        
        # Add grid lines to certainty map
        height, width = state.certainty_map.shape
        for i in range(height + 1):
            ax2.axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(width + 1):
            ax2.axvline(j - 0.5, color='black', linewidth=0.5)
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        for ax in [ax1, ax2]:
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()
    
    def _draw_grid(self, ax, grid: np.ndarray):
        """Draw the grid with appropriate colors"""
        height, width = grid.shape
        
        # Draw cells
        for i in range(height):
            for j in range(width):
                color = self.colors[grid[i, j]]
                rect = patches.Rectangle((j, height - i - 1), 1, 1, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor=color)
                ax.add_patch(rect)
                
                # Add X for unknown cells
                if grid[i, j] == -1:
                    ax.plot([j + 0.2, j + 0.8], [height - i - 0.2, height - i - 0.8], 
                           'k-', linewidth=1, alpha=0.5)
                    ax.plot([j + 0.2, j + 0.8], [height - i - 0.8, height - i - 0.2], 
                           'k-', linewidth=1, alpha=0.5)
        
        # Set limits and remove ticks
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _add_constraints(self, ax, row_constraints: List[List[int]], 
                        col_constraints: List[List[int]], grid_shape: Tuple[int, int]):
        """Add constraint numbers to the plot"""
        height, width = grid_shape
        
        # Row constraints (left side)
        max_row_constraint_len = max(len(c) for c in row_constraints) if row_constraints else 0
        for i, constraint in enumerate(row_constraints):
            constraint_str = ' '.join(map(str, constraint)) if constraint else '0'
            ax.text(-0.1, height - i - 0.5, constraint_str, 
                   ha='right', va='center', fontsize=9, fontweight='bold')
        
        # Column constraints (top)
        max_col_constraint_len = max(len(c) for c in col_constraints) if col_constraints else 0
        for j, constraint in enumerate(col_constraints):
            if constraint:
                for k, num in enumerate(constraint):
                    ax.text(j + 0.5, height + 0.1 + k * 0.3, str(num),
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(j + 0.5, height + 0.1, '0',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Adjust plot limits to accommodate constraints
        ax.set_xlim(-max_row_constraint_len * 0.5, width)
        ax.set_ylim(-0.5, height + max_col_constraint_len * 0.3 + 0.5)
    
    def save_plot(self, state: SolverState, filename: str, title: str = ""):
        """Save current plot to file"""
        fig, ax = plt.subplots(figsize=self.figsize)
        self._draw_grid(ax, state.grid)
        self._add_constraints(ax, state.row_constraints, state.col_constraints, state.grid.shape)
        
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold')
        
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_gif_frames(self, states: List[np.ndarray], 
                         constraints: Tuple[List[List[int]], List[List[int]]],
                         output_dir: str = "frames"):
        """Create individual frames for GIF animation"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        row_constraints, col_constraints = constraints
        
        for i, grid in enumerate(states):
            temp_state = type('TempState', (), {
                'grid': grid,
                'row_constraints': row_constraints,
                'col_constraints': col_constraints
            })()
            
            filename = os.path.join(output_dir, f"frame_{i:03d}.png")
            self.save_plot(temp_state, filename, f"Step {i + 1}")
        
        print(f"Created {len(states)} frames in {output_dir}/")

# Convenience functions
def quick_display(state: SolverState, title: str = "Nonogram"):
    """Quick display function for debugging"""
    displayer = GridDisplayer()
    displayer.display_puzzle(state, title)

def display_comparison(before: SolverState, after: SolverState):
    """Quick comparison display"""
    displayer = GridDisplayer()
    displayer.display_solution_comparison(before, after)