"""
Action Protocol for Semantic Parser Framework

This module defines the base protocol/interface that all actions must implement.
Actions are the building blocks of the ReACT-style reasoning loop.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import json


@dataclass
class ModuleConfig:
    """
    Configuration for a semantic parser module.
    
    Each module (e.g., Verdant, Cypher, Spider2) defines its own config
    that specifies target format, available actions, and other settings.
    """
    name: str
    target_format: str  # e.g., "SQL", "Cypher", "SPARQL"
    description: str = ""
    predecided_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActionInput(BaseModel):
    """Base model for action inputs with validation"""
    pass


class ActionOutput(BaseModel):
    """Base model for action outputs"""
    success: bool = Field(description="Whether the action executed successfully")
    result: Any = Field(description="The result of the action")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_string(self) -> str:
        """
        Serialize the action output to a string representation.
        
        Returns:
            String representation of the action result
        """
        if not self.success:
            return f"Error: {self.error}"
        
        # Handle different result types
        if isinstance(self.result, str):
            return self.result
        elif isinstance(self.result, (dict, list)):
            return json.dumps(self.result, indent=2)
        else:
            return str(self.result)

class Action(ABC):
    """
    Abstract base class for all actions in the framework.
    
    Each action represents a discrete operation that can be taken during
    the reasoning process (e.g., LLM call, database query, API call, search).
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize an action.
        
        Args:
            name: Unique identifier for the action
            description: Detailed description of what the action does, when to use it,
                        and what inputs it expects. This will be provided to the LLM.
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema for the action's input parameters.
        
        This schema is used to:
        1. Inform the LLM about what parameters are needed
        2. Validate inputs before execution
        
        Returns:
            A JSON schema dictionary describing the input format
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ActionOutput:
        """
        Execute the action with the given parameters.
        
        Args:
            **kwargs: Action-specific parameters matching the input schema
            
        Returns:
            ActionOutput containing the result and execution status
        """
        pass
    
    def get_action_spec(self) -> Dict[str, Any]:
        """
        Get the complete specification of this action for LLM prompting.
        
        Returns:
            Dictionary containing name, description, and input schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_input_schema()
        }
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters against the schema.
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Basic validation - can be extended with jsonschema library
        schema = self.get_input_schema()
        required = schema.get("required", [])
        
        for field in required:
            if field not in kwargs:
                raise ValueError(f"Missing required parameter: {field}")
        
        return True


class ActionRegistry:
    """
    Registry for managing all available actions.
    
    This allows dynamic registration of actions and provides
    a centralized way to access action specifications.
    
    The registry can also hold module configuration which defines
    the target format and other module-specific settings.
    """
    
    def __init__(self, module_config: Optional[ModuleConfig] = None):
        """
        Initialize the action registry.
        
        Args:
            module_config: Optional module configuration that defines
                          target format and predecided actions
        """
        self._actions: Dict[str, Action] = {}
        self._module_config = module_config
    
    @property
    def module_config(self) -> Optional[ModuleConfig]:
        """Get the module configuration"""
        return self._module_config
    
    @module_config.setter
    def module_config(self, config: ModuleConfig) -> None:
        """Set the module configuration"""
        self._module_config = config
    
    @property
    def target_format(self) -> Optional[str]:
        """Get the target format from module config"""
        return self._module_config.target_format if self._module_config else None
    
    @property
    def predecided_actions(self) -> List[str]:
        """Get predecided actions from module config"""
        return self._module_config.predecided_actions if self._module_config else []
    
    def register(self, action: Action) -> None:
        """
        Register an action in the registry.
        
        Args:
            action: Action instance to register
        """
        if action.name in self._actions:
            raise ValueError(f"Action '{action.name}' is already registered")
        self._actions[action.name] = action
    
    def get_action(self, name: str) -> Optional[Action]:
        """
        Retrieve an action by name.
        
        Args:
            name: Name of the action to retrieve
            
        Returns:
            Action instance or None if not found
        """
        return self._actions.get(name)
    
    def get_all_actions(self) -> Dict[str, Action]:
        """Get all registered actions"""
        return self._actions.copy()
    
    def get_action_specs(self) -> List[Dict[str, Any]]:
        """
        Get specifications for all registered actions.
        
        Returns:
            List of action specifications for LLM prompting
        """
        return [action.get_action_spec() for action in self._actions.values()]
    
    def unregister(self, name: str) -> None:
        """Remove an action from the registry"""
        if name in self._actions:
            del self._actions[name]
