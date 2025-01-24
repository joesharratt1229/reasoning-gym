"""
Game tasks for training reasoning capabilities:
- Board games
- Puzzle games
- Strategy games
"""

from .mini_sudoku import MiniSudokuConfig, MiniSudokuDataset, mini_sudoku_dataset
from .sudoku import SudokuConfig, SudokuDataset, sudoku_dataset

__all__ = [
    "MiniSudokuConfig",
    "MiniSudokuDataset",
    "mini_sudoku_dataset",
    "SudokuConfig",
    "SudokuDataset",
    "sudoku_dataset",
]
