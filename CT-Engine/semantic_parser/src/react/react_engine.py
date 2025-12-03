"""
ReACT Engine

This is the core engine that orchestrates the ReACT-style reasoning loop
for semantic parsing.
"""

import logging
from typing import Optional, Dict, Any, List
from semantic_parser.action_protocol import ActionRegistry, ActionOutput
from semantic_parser.llm_client import AzureOpenAIClient, Message
from semantic_parser.src.react.reasoning_step import ReasoningStep, ReasoningTrace, Thought, ActionCall, Observation
from semantic_parser.src.react.prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)


class ReACTEngine:
    """
    Main engine for ReACT-style semantic parsing.
    
    This engine:
    1. Maintains the reasoning trace
    2. Prompts the LLM for thoughts and actions
    3. Executes actions via the action registry
    4. Continues the loop until completion or max steps
    
    The target format is determined by the ActionRegistry's module configuration,
    so you don't need to specify it when calling parse().
    """
    
    def __init__(
        self,
        llm_client: AzureOpenAIClient,
        action_registry: ActionRegistry,
        max_steps: int = 10,
        verbose: bool = False,
        predecided_actions: Optional[List[str]] = None
    ):
        """
        Initialize the ReACT engine.
        
        Args:
            llm_client: Client for making LLM API calls
            action_registry: Registry of available actions (should have module_config set)
            max_steps: Maximum number of reasoning steps
            verbose: Whether to print detailed logs
            predecided_actions: Optional list of action names to execute before LLM-driven reasoning.
                              If not provided, uses predecided_actions from registry's module_config.
        """
        self.llm_client = llm_client
        self.action_registry = action_registry
        self.max_steps = max_steps
        self.verbose = verbose
        
        # Use predecided actions from argument, or fall back to registry's module config
        if predecided_actions is not None:
            self.predecided_actions = predecided_actions
        else:
            self.predecided_actions = action_registry.predecided_actions or []
        
        self.prompt_builder = PromptBuilder(action_registry)
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    @property
    def target_format(self) -> str:
        """Get the target format from the action registry's module config."""
        if self.action_registry.target_format:
            return self.action_registry.target_format
        return "SQL"  # Default fallback
    
    def parse(
        self,
        query: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """
        Parse a natural language query into a logical form.
        
        This is the main entry point that runs the complete ReACT loop.
        The target format is automatically determined from the action registry's
        module configuration.
        
        Args:
            query: Natural language query to parse
            initial_context: Optional initial context/metadata
            
        Returns:
            Complete ReasoningTrace with all steps and final output
        """
        target_format = self.target_format
        
        # Initialize the reasoning trace
        trace = ReasoningTrace(
            query=query,
            target_format=target_format,
            metadata=initial_context or {}
        )
        
        logger.info(f"Starting semantic parsing for query: {query}")
        logger.info(f"Target format: {target_format}")
        
        step_num = 1
        
        # Execute predecided actions first
        if self.predecided_actions:
            logger.info(f"Executing {len(self.predecided_actions)} predecided actions first")
            
            for action_name in self.predecided_actions:
                logger.info(f"\n{'='*60}")
                logger.info(f"Predecided Step {step_num}/{self.max_steps}: {action_name}")
                logger.info(f"{'='*60}")
                
                # Create a simple thought for predecided action
                thought = Thought(content=f"Executing predecided action: {action_name}")
                
                # Create action call with query as parameter
                action_call = ActionCall(
                    action_name=action_name,
                    parameters={"query": query}
                )
                
                if self.verbose:
                    print(f"\nPredecided Action: {action_name}")
                    print(f"Parameters: {action_call.parameters}")
                
                # Execute the action
                observation = self._execute_action(action_call)
                
                if self.verbose:
                    print(f"Observation: {observation.result}")
                    if not observation.success:
                        print(f"Error: {observation.error}")
                
                # Create and add the reasoning step
                step = ReasoningStep(
                    step_number=step_num,
                    thought=thought,
                    action=action_call,
                    observation=observation
                )
                trace.add_step(step)
                
                # Store observation result in metadata for later use
                if observation.success:
                    trace.metadata[f"{action_name}_result"] = observation.result
                
                step_num += 1
                
                # Check if we hit max steps
                if step_num > self.max_steps:
                    logger.warning("Reached maximum steps during predecided actions")
                    trace.metadata["completed"] = False
                    trace.metadata["reason"] = "max_steps_reached_during_predecided_actions"
                    return trace
        
        # Run the regular reasoning loop
        for _ in range(step_num, self.max_steps + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {step_num}/{self.max_steps}")
            logger.info(f"{'='*60}")
            
            # Get next thought and action from LLM
            thought, action_call = self._get_next_action(trace, step_num)
            
            if self.verbose:
                print(f"\nThought: {thought.content}")
                print(f"Action: {action_call.action_name}")
                print(f"Parameters: {action_call.parameters}")
            
            # Check if we're done
            if action_call.action_name == "finish" or action_call.action_name == "Finish":
                # Extract the result from the action parameters
                if "natural_language_response" in action_call.parameters:
                    final_output = action_call.parameters.get("natural_language_response", "")
                elif "output" in action_call.parameters:
                    final_output = action_call.parameters.get("output", "")
                else:
                    # Execute the finish action to get the natural language response
                    observation = self._execute_action(action_call)
                    if observation.success and isinstance(observation.result, dict):
                        final_output = observation.result.get("natural_language_response", str(observation.result))
                    else:
                        final_output = str(observation.result)
                
                trace.complete(final_output)
                logger.info(f"Reasoning complete. Final output: {final_output}")
                break
            
            # Execute the action
            observation = self._execute_action(action_call)
            
            if self.verbose:
                print(f"Observation: {observation.result}")
                if not observation.success:
                    print(f"Error: {observation.error}")
            
            # Create and add the reasoning step
            step = ReasoningStep(
                step_number=step_num,
                thought=thought,
                action=action_call,
                observation=observation
            )
            trace.add_step(step)
            
            # Check if we hit max steps
            if step_num >= self.max_steps:
                logger.warning("Reached maximum steps without completion")
                trace.metadata["completed"] = False
                trace.metadata["reason"] = "max_steps_reached"
                break
            
            step_num += 1
        
        return trace
    
    def _get_next_action(
        self,
        trace: ReasoningTrace,
        step_num: int
    ) -> tuple[Thought, ActionCall]:
        """
        Get the next thought and action from the LLM.
        
        Args:
            trace: Current reasoning trace
            step_num: Current step number
            
        Returns:
            Tuple of (Thought, ActionCall)
        """
        # Build the prompt
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_reasoning_prompt(trace, self.max_steps)
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]
        
        # Get response from LLM
        response = self.llm_client.chat_completion(messages)
        
        if self.verbose:
            print(f"\nLLM Response:\n{response.content}\n")
        
        # Parse the response
        parsed = self.prompt_builder.parse_llm_response(response.content)
        
        # Create thought and action objects
        thought = Thought(content=parsed["thought"])
        action_call = ActionCall(
            action_name=parsed["action"],
            parameters=parsed["parameters"]
        )
        
        return thought, action_call
    
    def _execute_action(self, action_call: ActionCall) -> Observation:
        """
        Execute an action and return the observation.
        
        Args:
            action_call: The action to execute
            
        Returns:
            Observation with the result
        """
        action_name = action_call.action_name
        parameters = action_call.parameters
        
        # Get the action from registry
        action = self.action_registry.get_action(action_name)
        
        if action is None:
            logger.error(f"Action not found: {action_name}")
            return Observation(
                action_name=action_name,
                result=None,
                success=False,
                error=f"Action '{action_name}' not found in registry"
            )
        
        try:
            # Validate input
            action.validate_input(**parameters)
            
            # Execute the action
            logger.info(f"Executing action: {action_name}")
            result = action.execute(**parameters)
            
            # Create observation
            observation = Observation(
                action_name=action_name,
                result=result.result,
                success=result.success,
                error=result.error,
                metadata=result.metadata
            )
            
            return observation
            
        except Exception as e:
            logger.error(f"Error executing action {action_name}: {str(e)}")
            return Observation(
                action_name=action_name,
                result=None,
                success=False,
                error=str(e)
            )
    
    def run_streaming(
        self,
        query: str,
        initial_context: Optional[Dict[str, Any]] = None
    ):
        """
        Run the parsing with streaming output for real-time visibility.
        
        This is a generator that yields reasoning steps as they happen.
        
        Args:
            query: Natural language query to parse
            initial_context: Optional initial context
            
        Yields:
            ReasoningStep objects as they are completed
        """
        target_format = self.target_format
        
        trace = ReasoningTrace(
            query=query,
            target_format=target_format,
            metadata=initial_context or {}
        )
        
        for step_num in range(1, self.max_steps + 1):
            thought, action_call = self._get_next_action(trace, step_num)
            
            if action_call.action_name == "finish":
                final_output = action_call.parameters.get("output", "")
                trace.complete(final_output)
                yield trace
                break
            
            observation = self._execute_action(action_call)
            
            step = ReasoningStep(
                step_number=step_num,
                thought=thought,
                action=action_call,
                observation=observation
            )
            trace.add_step(step)
            
            yield step
            
            if step_num >= self.max_steps:
                trace.metadata["completed"] = False
                yield trace
                break
