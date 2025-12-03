"""
Reasoning Step Model

This module defines the data structures for tracking each step
in the ReACT-style reasoning loop.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Thought(BaseModel):
    """Represents the LLM's reasoning/thinking at a step"""
    content: str = Field(description="The reasoning or thought process")
    timestamp: datetime = Field(default_factory=datetime.now)


class ActionCall(BaseModel):
    """Represents an action to be executed"""
    action_name: str = Field(description="Name of the action to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")
    timestamp: datetime = Field(default_factory=datetime.now)


class Observation(BaseModel):
    """Represents the result/observation from an action"""
    action_name: str = Field(description="Name of the action that was executed")
    result: Any = Field(description="The result from the action")
    success: bool = Field(description="Whether the action succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now)


class ReasoningStep(BaseModel):
    """
    Represents a complete reasoning step in the ReACT loop.
    
    Each step consists of:
    1. Thought: The LLM's reasoning about what to do
    2. Action: The action chosen to execute
    3. Observation: The result from executing the action
    """
    step_number: int = Field(description="Sequential step number")
    thought: Thought = Field(description="The reasoning process")
    action: ActionCall = Field(description="The action to execute")
    observation: Optional[Observation] = Field(default=None, description="Result from action execution")
    
    def is_complete(self) -> bool:
        """Check if this step has been fully executed"""
        return self.observation is not None


class ReasoningTrace(BaseModel):
    """
    Complete trace of the reasoning process.
    
    This tracks all steps from initial query to final answer.
    """
    query: str = Field(description="The original natural language query")
    target_format: str = Field(description="Target logical form (SQL, SPARQL, Cypher, etc.)")
    steps: List[ReasoningStep] = Field(default_factory=list, description="All reasoning steps")
    final_output: Optional[str] = Field(default=None, description="The final parsed logical form")
    is_complete: bool = Field(default=False, description="Whether reasoning is complete")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(default=None)
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the trace"""
        self.steps.append(step)
    
    def get_current_step(self) -> Optional[ReasoningStep]:
        """Get the most recent step"""
        return self.steps[-1] if self.steps else None
    
    def get_step_count(self) -> int:
        """Get the total number of steps"""
        return len(self.steps)
    
    def complete(self, final_output: str) -> None:
        """Mark the reasoning process as complete"""
        self.final_output = final_output
        self.is_complete = True
        self.end_time = datetime.now()
    
    def get_duration(self) -> Optional[float]:
        """Get the total duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_context_string(self) -> str:
        """
        Convert the reasoning trace to a string for LLM context.
        
        This creates a formatted string of all previous steps
        that can be included in the LLM prompt.
        """
        context_parts = [f"Query: {self.query}"]
        context_parts.append(f"Target Format: {self.target_format}")
        context_parts.append("\nReasoning History:")
        
        for step in self.steps:
            context_parts.append(f"\n--- Step {step.step_number} ---")
            context_parts.append(f"Thought: {step.thought.content}")
            context_parts.append(f"Action: {step.action.action_name}")
            context_parts.append(f"Parameters: {step.action.parameters}")
            
            if step.observation:
                context_parts.append(f"Observation: {step.observation.result}")
                if not step.observation.success:
                    context_parts.append(f"Error: {step.observation.error}")
        
        return "\n".join(context_parts)

