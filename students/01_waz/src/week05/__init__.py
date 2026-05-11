"""
Week 05: Seeing the Invisible - Covariance & Multicollinearity
--------------------------------------------------------------

This module implements Monte Carlo simulation for studying the effects
of multicollinearity on parameter estimation variance.

Author: 01_waz
Date: $(date '+%Y-%m-%d')
"""

from .data_generator import generate_design_matrix, generate_response, calculate_correlation
from .solvers import AnalyticalSolver
from .simulation import monte_carlo_simulation, run_comparison_experiments
from .analysis import create_scatter_comparison, print_covariance_comparison

__version__ = "1.0.0"
__author__ = "01_waz"
__all__ = [
    "generate_design_matrix", 
    "generate_response", 
    "calculate_correlation",
    "AnalyticalSolver",
    "monte_carlo_simulation", 
    "run_comparison_experiments",
    "create_scatter_comparison", 
    "print_covariance_comparison"
]
