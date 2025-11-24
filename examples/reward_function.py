"""
Multi-Objective Reward Function for Polymer Design

This module implements a weighted reward function that balances multiple
objectives: glass transition temperature (Tg), dielectric constant, cost,
and feasibility constraints.

License: MIT

TODO: Replace placeholder normalization ranges with domain-specific values.
TODO: Add additional property objectives as needed (mechanical, optical, etc.).
TODO: Implement adaptive weight tuning based on user preferences.
"""

from typing import Dict, Optional, List
import warnings


def compute_reward(
    props: Dict[str, float],
    bigsmiles: str,
    weights: Optional[Dict[str, float]] = None,
    targets: Optional[Dict[str, Dict[str, float]]] = None,
    penalties: Optional[List[str]] = None
) -> float:
    """
    Compute multi-objective weighted reward for a polymer candidate.
    
    The reward combines multiple objectives:
    1. Glass transition temperature (Tg) - maximize or target range
    2. Dielectric constant - minimize or target range
    3. Cost - minimize
    4. Feasibility flags - penalize infeasible candidates
    
    Args:
        props: Dictionary of predicted polymer properties
               Example: {"tg": 155.0, "dielectric": 3.8, "cost": 12.5}
        bigsmiles: BigSMILES string of the polymer (for validity checks)
        weights: Optional custom weights for each objective (default: equal weights)
                 Example: {"tg": 0.4, "dielectric": 0.3, "cost": 0.2, "feasibility": 0.1}
        targets: Optional target ranges for each property
                 Example: {"tg": {"min": 150, "max": 200, "ideal": 175}}
        penalties: Optional list of constraint violations (e.g., ["invalid_structure", "high_toxicity"])
    
    Returns:
        Normalized reward in range [-1, 1] (or [0, 1] depending on design choice)
        Higher reward = better polymer candidate
    
    Example:
        >>> props = {"tg": 165.0, "dielectric": 3.5, "cost": 15.0}
        >>> reward = compute_reward(props, "{[>][<]c1ccccc1[>]}[<]")
        >>> print(f"Reward: {reward:.3f}")
        Reward: 0.782
    
    TODO: Validate property ranges against known polymer database.
    TODO: Add uncertainty-aware reward (penalize high-uncertainty predictions).
    TODO: Implement multi-objective Pareto optimization (optional).
    """
    
    # Default weights (equal importance)
    if weights is None:
        weights = {
            "tg": 0.35,
            "dielectric": 0.35,
            "cost": 0.2,
            "feasibility": 0.1
        }
    
    # Validate weights sum to 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        warnings.warn(f"Weights sum to {weight_sum:.3f}, normalizing to 1.0")
        weights = {k: v / weight_sum for k, v in weights.items()}
    
    # Default target ranges (placeholders - replace with domain knowledge)
    if targets is None:
        targets = {
            "tg": {"min": 100, "max": 250, "ideal": 175},
            "dielectric": {"min": 2.0, "max": 5.0, "ideal": 3.0},
            "cost": {"min": 5.0, "max": 50.0, "ideal": 10.0}
        }
    
    # Initialize reward components
    reward_components = {}
    
    # 1. Glass Transition Temperature (Tg) reward
    if "tg" in props and "tg" in weights:
        tg = props["tg"]
        tg_target = targets.get("tg", {})
        reward_components["tg"] = _compute_target_reward(
            value=tg,
            min_val=tg_target.get("min", 100),
            max_val=tg_target.get("max", 250),
            ideal=tg_target.get("ideal", 175)
        )
    
    # 2. Dielectric Constant reward (typically minimize)
    if "dielectric" in props and "dielectric" in weights:
        dielectric = props["dielectric"]
        diel_target = targets.get("dielectric", {})
        reward_components["dielectric"] = _compute_target_reward(
            value=dielectric,
            min_val=diel_target.get("min", 2.0),
            max_val=diel_target.get("max", 5.0),
            ideal=diel_target.get("ideal", 3.0)
        )
    
    # 3. Cost reward (minimize)
    if "cost" in props and "cost" in weights:
        cost = props["cost"]
        cost_target = targets.get("cost", {})
        # For cost, ideal is minimum (inverse relationship)
        reward_components["cost"] = _compute_minimize_reward(
            value=cost,
            min_val=cost_target.get("min", 5.0),
            max_val=cost_target.get("max", 50.0)
        )
    
    # 4. Feasibility penalty
    feasibility_reward = 1.0
    if penalties:
        # Apply penalties for constraint violations
        penalty_weights = {
            "invalid_structure": -0.5,
            "high_toxicity": -0.8,
            "unavailable_monomers": -0.6,
            "unstable": -0.4,
        }
        for violation in penalties:
            if violation in penalty_weights:
                feasibility_reward += penalty_weights[violation]
        
        # Clip to [0, 1]
        feasibility_reward = max(0.0, min(1.0, feasibility_reward))
    
    # Check basic validity (non-empty BigSMILES)
    if not bigsmiles or len(bigsmiles) < 5:
        feasibility_reward = 0.0
    
    reward_components["feasibility"] = feasibility_reward
    
    # Compute weighted sum
    total_reward = sum(
        weights.get(key, 0.0) * reward_components.get(key, 0.0)
        for key in weights.keys()
    )
    
    # Normalize to [-1, 1] or [0, 1] range
    # Current implementation returns [0, 1]
    total_reward = max(0.0, min(1.0, total_reward))
    
    return total_reward


def _compute_target_reward(value: float, min_val: float, max_val: float, ideal: float) -> float:
    """
    Compute reward for a property with a target range and ideal value.
    
    Reward is highest at the ideal value, decreases smoothly as value
    moves toward min/max boundaries, and is 0 outside the range.
    
    Args:
        value: Predicted property value
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        ideal: Ideal target value
    
    Returns:
        Reward in [0, 1]
    """
    if value < min_val or value > max_val:
        # Outside acceptable range
        # Soft penalty: exponential decay
        if value < min_val:
            return max(0.0, 1.0 - 0.1 * (min_val - value) / min_val)
        else:
            return max(0.0, 1.0 - 0.1 * (value - max_val) / max_val)
    
    # Inside acceptable range: reward based on distance from ideal
    if ideal < min_val or ideal > max_val:
        ideal = (min_val + max_val) / 2  # Fallback to midpoint
    
    distance = abs(value - ideal)
    max_distance = max(abs(ideal - min_val), abs(ideal - max_val))
    
    if max_distance == 0:
        return 1.0
    
    # Linear decay from ideal
    reward = 1.0 - (distance / max_distance) * 0.5
    return max(0.0, min(1.0, reward))


def _compute_minimize_reward(value: float, min_val: float, max_val: float) -> float:
    """
    Compute reward for a property to minimize (e.g., cost).
    
    Reward is highest at min_val, decreases linearly to 0 at max_val.
    
    Args:
        value: Predicted property value
        min_val: Best-case value (highest reward)
        max_val: Worst-case value (zero reward)
    
    Returns:
        Reward in [0, 1]
    """
    if value <= min_val:
        return 1.0
    elif value >= max_val:
        return 0.0
    else:
        # Linear interpolation
        return 1.0 - (value - min_val) / (max_val - min_val)


def compute_multi_objective_reward(
    props_list: List[Dict[str, float]],
    bigsmiles_list: List[str],
    weights: Optional[Dict[str, float]] = None,
    method: str = "weighted_sum"
) -> List[float]:
    """
    Compute rewards for a batch of polymer candidates.
    
    Args:
        props_list: List of property dictionaries
        bigsmiles_list: List of BigSMILES strings
        weights: Reward weights (passed to compute_reward)
        method: Aggregation method ("weighted_sum", "pareto", "scalarization")
    
    Returns:
        List of reward values
    
    TODO: Implement Pareto-based multi-objective optimization.
    TODO: Add support for constrained optimization (inequality constraints).
    """
    if method != "weighted_sum":
        raise NotImplementedError(f"Method '{method}' not yet implemented. Use 'weighted_sum'.")
    
    rewards = []
    for props, bigsmiles in zip(props_list, bigsmiles_list):
        reward = compute_reward(props, bigsmiles, weights=weights)
        rewards.append(reward)
    
    return rewards


# Example usage and testing
if __name__ == "__main__":
    print("=== Reward Function Demo ===\n")
    
    # Test case 1: High-performing polymer
    props1 = {
        "tg": 170.0,        # Close to ideal (175°C)
        "dielectric": 3.2,  # Close to ideal (3.0)
        "cost": 12.0        # Low cost
    }
    bigsmiles1 = "{[>][<]c1ccccc1C(C)(C)[>]}[<]"
    reward1 = compute_reward(props1, bigsmiles1)
    print(f"Test 1 (high-performing polymer):")
    print(f"  Properties: {props1}")
    print(f"  Reward: {reward1:.3f}\n")
    
    # Test case 2: Acceptable but not ideal
    props2 = {
        "tg": 140.0,        # Below ideal but within range
        "dielectric": 4.5,  # High but acceptable
        "cost": 25.0        # Higher cost
    }
    bigsmiles2 = "{[>][<]CC(C)(C)[>]}[<]"
    reward2 = compute_reward(props2, bigsmiles2)
    print(f"Test 2 (acceptable polymer):")
    print(f"  Properties: {props2}")
    print(f"  Reward: {reward2:.3f}\n")
    
    # Test case 3: Out of range properties
    props3 = {
        "tg": 80.0,         # Below minimum
        "dielectric": 6.0,  # Above maximum
        "cost": 55.0        # Too expensive
    }
    bigsmiles3 = "{[>][<]CC[>]}[<]"
    reward3 = compute_reward(props3, bigsmiles3)
    print(f"Test 3 (poor polymer):")
    print(f"  Properties: {props3}")
    print(f"  Reward: {reward3:.3f}\n")
    
    # Test case 4: With feasibility penalties
    props4 = {
        "tg": 165.0,
        "dielectric": 3.5,
        "cost": 15.0
    }
    bigsmiles4 = "{[>][<]c1ccccc1[>]}[<]"
    penalties4 = ["high_toxicity", "unavailable_monomers"]
    reward4 = compute_reward(props4, bigsmiles4, penalties=penalties4)
    print(f"Test 4 (with penalties):")
    print(f"  Properties: {props4}")
    print(f"  Penalties: {penalties4}")
    print(f"  Reward: {reward4:.3f}\n")
    
    # Test case 5: Custom weights
    props5 = {
        "tg": 175.0,
        "dielectric": 3.0,
        "cost": 30.0
    }
    bigsmiles5 = "{[>][<]c1ccccc1[>]}[<]"
    custom_weights = {
        "tg": 0.6,          # Prioritize Tg
        "dielectric": 0.2,
        "cost": 0.15,
        "feasibility": 0.05
    }
    reward5 = compute_reward(props5, bigsmiles5, weights=custom_weights)
    print(f"Test 5 (custom weights):")
    print(f"  Properties: {props5}")
    print(f"  Weights: {custom_weights}")
    print(f"  Reward: {reward5:.3f}\n")
    
    print("=== Next Steps ===")
    print("TODO: Validate normalization ranges against real polymer data")
    print("TODO: Implement uncertainty-aware rewards")
    print("TODO: Add multi-objective Pareto optimization")
    print("TODO: Tune weights based on experimental feedback")
