"""
Week 04: Solvers Showdown (求解器双城记)
-----------------------------------------

This module implements two solvers for linear regression:
1. AnalyticalSolver: Closed-form solution using normal equations
2. GradientDescentSolver: Iterative optimization using gradient descent

Author: 01_waz
Date: $(date '+%Y-%m-%d')
"""

from .solvers import AnalyticalSolver, GradientDescentSolver

__version__ = "1.0.0"
__author__ = "01_waz"
__all__ = ["AnalyticalSolver", "GradientDescentSolver"]
