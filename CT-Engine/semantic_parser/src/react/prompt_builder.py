"""
Prompt Builder for ReACT-style Reasoning

This module constructs prompts for the LLM at each reasoning step,
including system instructions, available actions, and reasoning history.
"""

from typing import List, Dict, Any
from semantic_parser.src.react.reasoning_step import ReasoningTrace
from semantic_parser.action_protocol import ActionRegistry


class PromptBuilder:
    """
    Builds prompts for the LLM at each reasoning step.
    
    The prompt includes:
    1. System instructions for ReACT-style reasoning
    2. Available actions with their specifications
    3. Previous reasoning history
    4. Instructions for the current step
    """
    
    def __init__(self, action_registry: ActionRegistry):
        """
        Initialize the prompt builder.
        
        Args:
            action_registry: Registry containing all available actions
        """
        self.action_registry = action_registry
    
    def build_system_prompt(self) -> str:
        """
        Build the system prompt explaining the ReACT reasoning process.
        
        Returns:
            System prompt string
        """
        system_prompt = """You are an expert Database QA System based on semantic parsing that converts natural language queries into formal logical representations (SQL, SPARQL, Cypher, etc.). Given a query, your final goal is to return the correct answer by parsing the query into a logical form and executing it against the relevant database.

You operate using a ReACT (Reasoning + Acting) framework. At each step, you:

1. **Think**: Reason about the current situation and what needs to be done next
2. **Act**: Choose and execute an action from the available actions
3. **Observe**: Receive the result of your action and use it for the next step

Your response at each step MUST follow this exact format:

Thought: [Your reasoning about what to do next and why]
Action: [The exact name of the action to execute]
Action Input: [A JSON object with the parameters for the action]

Example:
Thought: I need to understand the database schema before writing a SQL query. Let me retrieve the schema information.
Action: get_schema
Action Input: {"database": "employees"}

Continue this process until you have obtained the correct answer from the generated logical form. When you're ready to provide the final answer, use:

Thought: [Your final reasoning]
Action: finish
Action Input: {"output": "[The final table from the executed query]"}

Important guidelines:
- Always explain your reasoning in the Thought section
- Choose actions that make logical progress toward the goal
- Use observations from previous steps to inform your decisions
- Be systematic and thorough in your approach
- The Action Input must be valid JSON
"""
        return system_prompt
    
    def build_actions_description(self) -> str:
        """
        Build a description of all available actions.
        
        Returns:
            Formatted string describing all actions
        """
        action_specs = self.action_registry.get_action_specs()
        
        if not action_specs:
            return "No actions available."
        
        descriptions = ["Available Actions:", ""]
        
        for spec in action_specs:
            descriptions.append(f"Action: {spec['name']}")
            descriptions.append(f"Description: {spec['description']}")
            descriptions.append(f"Input Schema: {spec['input_schema']}")
            descriptions.append("")
        
        return "\n".join(descriptions)
    
    def build_reasoning_prompt(
        self,
        trace: ReasoningTrace,
        max_steps: int = 10
    ) -> str:
        """
        Build the complete prompt for the next reasoning step.
        
        Args:
            trace: Current reasoning trace with history
            max_steps: Maximum number of steps allowed
            
        Returns:
            Complete prompt string for the LLM
        """
        prompt_parts = []
        
        # Add the current task
        prompt_parts.append(f"Task: Convert the following natural language query into {trace.target_format}")
        prompt_parts.append(f"Query: {trace.query}")
        prompt_parts.append("")
        
        # Add available actions
        prompt_parts.append(self.build_actions_description())
        
        # Add reasoning history if exists
        if trace.steps:
            prompt_parts.append("Previous Steps:")
            prompt_parts.append("")
            
            for step in trace.steps:
                prompt_parts.append(f"Step {step.step_number}:")
                prompt_parts.append(f"Thought: {step.thought.content}")
                prompt_parts.append(f"Action: {step.action.action_name}")
                prompt_parts.append(f"Action Input: {step.action.parameters}")
                
                if step.observation:
                    if step.observation.success:
                        prompt_parts.append(f"Observation: {step.observation.result}")
                    else:
                        prompt_parts.append(f"Observation: Action failed - {step.observation.error}")
                
                prompt_parts.append("")
        
        # Add step count warning
        current_step = trace.get_step_count() + 1
        prompt_parts.append(f"Current Step: {current_step}/{max_steps}")
        
        if current_step >= max_steps:
            prompt_parts.append("WARNING: You are at the maximum number of steps. Please provide a final answer.")
        
        prompt_parts.append("")
        prompt_parts.append("What is your next thought and action?")
        
        return "\n".join(prompt_parts)
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM's response to extract thought, action, and parameters.
        
        Expected format:
        Thought: [reasoning]
        Action: [action_name]
        Action Input: [json_parameters]
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Dictionary with 'thought', 'action', and 'parameters'
        """
        lines = response.strip().split('\n')
        
        thought = ""
        action = ""
        action_input = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Thought:"):
                current_section = "thought"
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                current_section = "action"
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                current_section = "action_input"
                action_input = line.replace("Action Input:", "").strip()
            elif current_section and line:
                # Continue the current section
                if current_section == "thought":
                    thought += " " + line
                elif current_section == "action_input":
                    action_input += " " + line
        
        # Parse JSON from action input
        import json
        try:
            parameters = json.loads(action_input) if action_input else {}
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as a single string parameter
            parameters = {"input": action_input}
        
        return {
            "thought": thought,
            "action": action,
            "parameters": parameters
        }

