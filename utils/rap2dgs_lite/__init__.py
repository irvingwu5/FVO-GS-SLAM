"""
RAP2DGS Lite: lightweight rule-based scoring for frozen handoff selection.

First version: candidate_mask + rule scoring + top K selection.
Only used for frozen handoff selection; never prunes active GaussianModel.
"""

from .scorer import RAP2DGSLiteScorer
from .selector import RAP2DGSLiteSelector

__all__ = ["RAP2DGSLiteScorer", "RAP2DGSLiteSelector"]
