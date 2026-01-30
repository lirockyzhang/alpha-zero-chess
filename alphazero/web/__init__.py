"""Web interface module for AlphaZero.

Provides Flask-based web interface for playing against trained models.
"""

from .app import ChessWebInterface

__all__ = ['ChessWebInterface']
