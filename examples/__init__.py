"""
AgentEvolver Polymer Design Integration Examples

This package provides example implementations and scaffolding for integrating
modelscope/AgentEvolver into polymer materials autonomous design workflows.

Modules:
    - gym_env: OpenAI Gym environment wrapper for polymer design
    - mutation_operators: Domain-aware polymer structure mutation functions
    - reward_function: Multi-objective reward computation
    - llm_planner: LLM adapter for high-level design planning

License: MIT
"""

__version__ = "0.1.0"
__author__ = "AgentEvolver Integration Team"

# Make key classes available at package level
try:
    from .gym_env import PolymerSurrogateEnv, DummySurrogatePredictor
    from .reward_function import compute_reward, compute_multi_objective_reward
    from .llm_planner import LLMPlannerAdapter
    from .mutation_operators import (
        mutate_substitute,
        mutate_block_length,
        mutate_composition,
        mutate_processing,
        combine_mutations
    )
    
    __all__ = [
        'PolymerSurrogateEnv',
        'DummySurrogatePredictor',
        'compute_reward',
        'compute_multi_objective_reward',
        'LLMPlannerAdapter',
        'mutate_substitute',
        'mutate_block_length',
        'mutate_composition',
        'mutate_processing',
        'combine_mutations',
    ]
except ImportError:
    # Allow package to be imported even if dependencies are missing
    __all__ = []
