"""
LLM Planner Adapter for Polymer Design

This module provides a minimal adapter for using Large Language Models (LLMs)
to generate high-level design plans and suggestions for polymer optimization.
The planner can guide the RL agent's exploration strategy.

License: MIT

TODO: Add support for multiple LLM providers (OpenAI, Anthropic, local models).
TODO: Implement prompt engineering and few-shot examples.
TODO: Add caching and rate limiting for API calls.
TODO: Handle API key management securely (environment variables, secrets manager).
"""

import json
import os
from typing import Dict, List, Optional, Any
import warnings


class LLMPlannerAdapter:
    """
    Adapter for LLM-based planning in polymer design workflows.
    
    Uses LLMs to provide high-level design strategies, such as:
    - Suggesting which monomers to substitute
    - Recommending structural modifications to achieve target properties
    - Explaining trade-offs between competing objectives
    
    Usage:
        >>> planner = LLMPlannerAdapter(api_key="your-api-key")
        >>> plan = planner.generate_plan(
        ...     current_bigsmiles="{[>][<]CC(C)(C)[>]}[<]",
        ...     current_properties={"tg": 120, "dielectric": 4.0},
        ...     target_properties={"tg": 160, "dielectric": 3.5}
        ... )
        >>> print(plan["suggestions"])
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the LLM Planner Adapter.
        
        Args:
            api_key: API key for LLM provider (if None, reads from environment)
            model: Model identifier (e.g., "gpt-4", "claude-3", "llama-2")
            provider: LLM provider ("openai", "anthropic", "huggingface")
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in response
        
        TODO: Implement secure API key management.
        """
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key is None:
            warnings.warn(
                "No API key provided. LLM calls will fail. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.api_key = api_key
        
        # Initialize client (placeholder - replace with actual SDK)
        self.client = None
        if self.provider == "openai" and self.api_key:
            try:
                from openai import OpenAI
                # Use OpenAI v1.0+ client initialization
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                warnings.warn("OpenAI SDK not installed. Install with: pip install openai")
    
    def generate_plan(
        self,
        current_bigsmiles: str,
        current_properties: Dict[str, float],
        target_properties: Dict[str, float],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a design plan using LLM.
        
        Args:
            current_bigsmiles: Current polymer BigSMILES string
            current_properties: Current predicted properties
            target_properties: Desired target properties
            context: Optional additional context or constraints
        
        Returns:
            Dictionary containing:
                - suggestions: List of suggested actions/modifications
                - reasoning: Explanation of the plan
                - priority: Priority order for suggestions
        
        Example:
            >>> plan = planner.generate_plan(
            ...     current_bigsmiles="{[>][<]CC(C)(C)[>]}[<]",
            ...     current_properties={"tg": 120},
            ...     target_properties={"tg": 160}
            ... )
            >>> print(plan["suggestions"])
            ["Add aromatic monomers to increase rigidity", "Increase block length"]
        """
        # Construct prompt
        prompt = self._build_prompt(
            current_bigsmiles,
            current_properties,
            target_properties,
            context
        )
        
        # Call LLM
        if self.client is None:
            # Fallback: Return mock suggestions
            warnings.warn("LLM client not initialized. Returning mock suggestions.")
            return self._mock_plan(current_properties, target_properties)
        
        try:
            response = self._call_llm(prompt)
            plan = self._parse_response(response)
            return plan
        except Exception as e:
            warnings.warn(f"LLM call failed: {e}. Returning mock suggestions.")
            return self._mock_plan(current_properties, target_properties)
    
    def _build_prompt(
        self,
        bigsmiles: str,
        current_props: Dict[str, float],
        target_props: Dict[str, float],
        context: Optional[str]
    ) -> str:
        """
        Build prompt for LLM based on current state and goals.
        
        Args:
            bigsmiles: Current polymer structure
            current_props: Current properties
            target_props: Target properties
            context: Optional context
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert polymer chemist and materials scientist. Your task is to suggest design modifications to a polymer to achieve target properties.

Current Polymer Structure (BigSMILES):
{bigsmiles}

Current Properties:
{json.dumps(current_props, indent=2)}

Target Properties:
{json.dumps(target_props, indent=2)}
"""
        
        if context:
            prompt += f"\nAdditional Context:\n{context}\n"
        
        prompt += """
Based on polymer chemistry principles, suggest 2-4 specific modifications to achieve the target properties. Consider:
- Monomer substitutions (e.g., adding aromatic groups to increase Tg)
- Block length adjustments (molecular weight effects)
- Compositional changes (copolymer ratios)
- Processing conditions (if applicable)

Provide your response as a JSON object with the following structure:
{
  "suggestions": [
    {
      "action": "substitute_monomer",
      "details": "Replace aliphatic segments with aromatic monomers",
      "reasoning": "Aromatic rings increase rigidity and Tg",
      "priority": 1
    },
    ...
  ],
  "overall_strategy": "Brief summary of the approach",
  "expected_impact": "Expected property changes"
}

Respond ONLY with valid JSON, no additional text.
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API with the prompt.
        
        Args:
            prompt: Input prompt
        
        Returns:
            LLM response text
        
        TODO: Implement retry logic and error handling.
        TODO: Add rate limiting and cost tracking.
        """
        if self.provider == "openai" and self.client:
            try:
                # Use OpenAI v1.0+ API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert polymer chemist."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"OpenAI API call failed: {e}")
        else:
            raise NotImplementedError(f"Provider '{self.provider}' not yet implemented")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured plan.
        
        Args:
            response: Raw LLM response text
        
        Returns:
            Parsed plan dictionary
        """
        try:
            # Try to extract JSON from response
            # Handle cases where LLM includes markdown code blocks
            response = response.strip()
            if "```json" in response:
                # Extract JSON from markdown code block
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            
            plan = json.loads(response)
            
            # Validate structure
            if "suggestions" not in plan:
                plan["suggestions"] = []
            if "overall_strategy" not in plan:
                plan["overall_strategy"] = "No strategy provided"
            if "expected_impact" not in plan:
                plan["expected_impact"] = "Unknown"
            
            return plan
        
        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to parse LLM response as JSON: {e}")
            # Fallback: Extract suggestions as plain text
            return {
                "suggestions": [{"action": "unknown", "details": response, "priority": 1}],
                "overall_strategy": "Parsing failed, see raw response",
                "expected_impact": "Unknown"
            }
    
    def _mock_plan(
        self,
        current_props: Dict[str, float],
        target_props: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate mock suggestions when LLM is unavailable.
        
        Uses simple heuristics based on property gaps.
        
        Args:
            current_props: Current properties
            target_props: Target properties
        
        Returns:
            Mock plan dictionary
        """
        suggestions = []
        
        # Heuristic: If Tg is below target, suggest adding aromatic groups
        if "tg" in target_props and "tg" in current_props:
            tg_gap = target_props["tg"] - current_props["tg"]
            if tg_gap > 10:
                suggestions.append({
                    "action": "substitute_monomer",
                    "details": "Add aromatic monomers to increase rigidity",
                    "reasoning": f"Tg is {tg_gap:.1f}°C below target. Aromatic groups increase Tg.",
                    "priority": 1
                })
        
        # Heuristic: If dielectric is above target, suggest reducing polarity
        if "dielectric" in target_props and "dielectric" in current_props:
            diel_gap = current_props["dielectric"] - target_props["dielectric"]
            if diel_gap > 0.5:
                suggestions.append({
                    "action": "substitute_monomer",
                    "details": "Replace polar groups with non-polar alternatives",
                    "reasoning": f"Dielectric constant is {diel_gap:.2f} above target.",
                    "priority": 2
                })
        
        # Heuristic: If cost is above target, suggest cheaper monomers
        if "cost" in target_props and "cost" in current_props:
            cost_gap = current_props["cost"] - target_props["cost"]
            if cost_gap > 5.0:
                suggestions.append({
                    "action": "substitute_monomer",
                    "details": "Use commercially available monomers",
                    "reasoning": f"Cost is ${cost_gap:.2f}/kg above target.",
                    "priority": 3
                })
        
        if not suggestions:
            suggestions.append({
                "action": "explore",
                "details": "Properties are close to target. Try minor variations.",
                "reasoning": "Current polymer is near optimal.",
                "priority": 1
            })
        
        return {
            "suggestions": suggestions,
            "overall_strategy": "Heuristic-based suggestions (LLM unavailable)",
            "expected_impact": "Estimated based on domain knowledge"
        }
    
    def generate_batch_plans(
        self,
        states: List[Dict[str, Any]],
        max_parallel: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate plans for multiple polymer states in parallel.
        
        Args:
            states: List of state dictionaries (bigsmiles, properties, targets)
            max_parallel: Maximum parallel API calls
        
        Returns:
            List of plan dictionaries
        
        TODO: Implement parallel API calls with rate limiting.
        """
        plans = []
        for state in states:
            plan = self.generate_plan(
                current_bigsmiles=state["bigsmiles"],
                current_properties=state["current_properties"],
                target_properties=state["target_properties"],
                context=state.get("context")
            )
            plans.append(plan)
        
        return plans


# Example usage and testing
if __name__ == "__main__":
    print("=== LLM Planner Adapter Demo ===\n")
    
    # Initialize planner (will use mock mode if no API key)
    planner = LLMPlannerAdapter(
        api_key=None,  # Set to your API key or use environment variable
        model="gpt-4",
        temperature=0.7
    )
    
    # Test case 1: Increase Tg
    print("Test 1: Increase Tg\n")
    plan1 = planner.generate_plan(
        current_bigsmiles="{[>][<]CC(C)(C)[>]}[<]",
        current_properties={"tg": 120.0, "dielectric": 3.8, "cost": 12.0},
        target_properties={"tg": 160.0, "dielectric": 3.5, "cost": 15.0}
    )
    print(f"Suggestions: {json.dumps(plan1['suggestions'], indent=2)}")
    print(f"Strategy: {plan1['overall_strategy']}")
    print(f"Expected Impact: {plan1['expected_impact']}\n")
    
    # Test case 2: Reduce dielectric constant
    print("Test 2: Reduce Dielectric Constant\n")
    plan2 = planner.generate_plan(
        current_bigsmiles="{[>][<]c1ccccc1[>]}[<]",
        current_properties={"tg": 165.0, "dielectric": 5.0, "cost": 18.0},
        target_properties={"tg": 160.0, "dielectric": 3.5, "cost": 20.0}
    )
    print(f"Suggestions: {json.dumps(plan2['suggestions'], indent=2)}")
    print(f"Strategy: {plan2['overall_strategy']}\n")
    
    # Test case 3: Reduce cost
    print("Test 3: Reduce Cost\n")
    plan3 = planner.generate_plan(
        current_bigsmiles="{[>][<]c1ccccc1C(C)(C)[>]}[<]",
        current_properties={"tg": 170.0, "dielectric": 3.5, "cost": 35.0},
        target_properties={"tg": 165.0, "dielectric": 3.5, "cost": 20.0}
    )
    print(f"Suggestions: {json.dumps(plan3['suggestions'], indent=2)}")
    print(f"Strategy: {plan3['overall_strategy']}\n")
    
    print("=== Next Steps ===")
    print("TODO: Set OPENAI_API_KEY environment variable to enable real LLM calls")
    print("TODO: Implement retry logic and error handling for API calls")
    print("TODO: Add few-shot examples in prompts for better suggestions")
    print("TODO: Integrate with AgentEvolver's planning module")
    print("TODO: Evaluate LLM suggestions against experimental data")
