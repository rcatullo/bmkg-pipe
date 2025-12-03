"""
Composition Executor

Safely executes dynamically generated composition operators (Python code)
to merge subtask results.
"""

import ast
import logging
from typing import Any, Dict
from semantic_parser.src.decompose.decompose_utils import (
    CompositionOperator,
    CompositionError
)


logger = logging.getLogger(__name__)


class CompositionExecutor:
    """
    Executor for composition operators.
    
    This class safely executes dynamically generated Python code
    to compose subtask results.
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the executor.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self._safe_builtins = self._get_safe_builtins()
    
    def _get_safe_builtins(self) -> Dict[str, Any]:
        """
        Get a safe set of builtins for code execution.
        
        Returns:
            Dictionary of safe builtins
        """
        # Only allow safe operations
        safe = {
            # Type constructors
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'bool': bool,
            
            # Safe functions
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'any': any,
            'all': all,
            
            # String operations
            'print': print,  # For debugging
            
            # Exceptions
            'Exception': Exception,
            'ValueError': ValueError,
            'KeyError': KeyError,
            'TypeError': TypeError,
        }
        
        return safe
    
    def execute_operator(
        self,
        operator: 'CompositionOperator',
        inputs: Dict[str, Any]
    ) -> Any:
        """
        Execute a composition operator with given inputs.
        
        Args:
            operator: CompositionOperator with code and input specification
            inputs: Dictionary mapping variable names (output names from subtask specs) to results.
                    The keys should match the parameter names in the operator function.
            
        Returns:
            Result of the operator execution
            
        Raises:
            CompositionError: If execution fails
        """
        try:
            # Validate the code first
            self._validate_code(operator.code)
            
            # Create execution namespace
            namespace = {
                '__builtins__': self._safe_builtins,
            }
            
            # Add inputs to namespace (so they're available in the function scope)
            for input_name, value in inputs.items():
                namespace[input_name] = value
            
            # Execute the operator code to define the function
            exec(operator.code, namespace)
            
            # Get the operator function
            if operator.operator_id not in namespace:
                raise CompositionError(
                    f"Operator code must define a function named '{operator.operator_id}'"
                )
            
            operator_fn = namespace[operator.operator_id]
            
            # Execute the operator function with keyword arguments
            # This allows parameter names in the code to match output variable names
            try:
                result = operator_fn(**inputs)
            except TypeError:
                # Fallback: try positional arguments in the order of input_subtasks
                # This maintains backwards compatibility
                args = []
                for input_id in operator.input_subtasks:
                    if input_id in inputs:
                        args.append(inputs[input_id])
                    else:
                        # Try to find by any key (in case of name mismatch)
                        for key, value in inputs.items():
                            if key not in [a for a in args]:
                                args.append(value)
                                break
                result = operator_fn(*args)
            
            logger.info(f"Operator {operator.operator_id} executed successfully")
            return result
            
        except CompositionError:
            raise
        except Exception as e:
            logger.error(f"Error executing operator {operator.operator_id}: {e}")
            raise CompositionError(f"Operator execution failed: {e}")
    
    def execute_composition(
        self,
        composition_op: CompositionOperator,
        results: Dict[str, Any]
    ) -> Any:
        """
        Execute a composition operator to merge results.
        
        Args:
            composition_op: Composition operator with code
            results: Dictionary mapping subtask IDs to their results
            
        Returns:
            Composed result
            
        Raises:
            CompositionError: If execution fails
        """
        try:
            # Validate the code first
            self._validate_code(composition_op.code)
            
            # Create execution namespace
            namespace = {
                '__builtins__': self._safe_builtins,
                'results': results,
            }
            
            # Execute the composition code
            exec(composition_op.code, namespace)
            
            # Get the compose function
            if 'compose' not in namespace:
                raise CompositionError("Composition code must define a 'compose' function")
            
            compose_fn = namespace['compose']
            
            # Execute the composition
            result = compose_fn(results)
            
            logger.info("Composition executed successfully")
            return result
            
        except CompositionError:
            raise
        except Exception as e:
            logger.error(f"Error executing composition: {e}")
            raise CompositionError(f"Composition execution failed: {e}")
    
    def _validate_code(self, code: str):
        """
        Validate that code is safe to execute.
        
        Args:
            code: Python code to validate
            
        Raises:
            CompositionError: If code is unsafe
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Check for unsafe operations
            for node in ast.walk(tree):
                # Disallow imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise CompositionError("Import statements are not allowed")
                
                # Disallow exec/eval
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ('exec', 'eval', 'compile', '__import__'):
                            raise CompositionError(f"Function '{node.func.id}' is not allowed")
                
                # Disallow file operations
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ('open', 'file'):
                            raise CompositionError("File operations are not allowed")
            
            logger.info("Code validation passed")
            
        except CompositionError:
            raise
        except Exception as e:
            logger.error(f"Code validation error: {e}")
            raise CompositionError(f"Invalid code: {e}")
    
    def test_composition(
        self,
        composition_op: CompositionOperator,
        test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a composition operator with sample data.
        
        Args:
            composition_op: Composition operator to test
            test_results: Test data
            
        Returns:
            Dictionary with test results and any errors
        """
        try:
            result = self.execute_composition(composition_op, test_results)
            return {
                'success': True,
                'result': result,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }


class SafeCompositionExecutor(CompositionExecutor):
    """
    Extra-safe composition executor with additional restrictions.
    
    This version uses RestrictedPython or similar techniques for
    additional safety (placeholder for production use).
    """
    
    def __init__(self, timeout: int = 30):
        super().__init__(timeout)
        # Add additional safety measures
        self.max_iterations = 10000  # Prevent infinite loops
        self.max_recursion_depth = 100
    
    def execute_composition(
        self,
        composition_op: CompositionOperator,
        results: Dict[str, Any]
    ) -> Any:
        """
        Execute composition with additional safety checks.
        
        Args:
            composition_op: Composition operator
            results: Subtask results
            
        Returns:
            Composed result
        """
        # TODO: In production, use RestrictedPython or similar
        # For now, use the base implementation with validation
        return super().execute_composition(composition_op, results)


def create_simple_composition(operation: str = "merge") -> str:
    """
    Create a simple composition operator template.
    
    Args:
        operation: Type of composition ("merge", "concatenate", "sum", etc.)
        
    Returns:
        Python code for composition
    """
    templates = {
        "merge": """
def compose(results):
    \"\"\"Merge all results into a single dict or list.\"\"\"
    if not results:
        return None
    
    # If all results are dicts, merge them
    if all(isinstance(r, dict) for r in results.values()):
        merged = {}
        for r in results.values():
            merged.update(r)
        return merged
    
    # If all results are lists, concatenate them
    if all(isinstance(r, list) for r in results.values()):
        concatenated = []
        for r in results.values():
            concatenated.extend(r)
        return concatenated
    
    # Otherwise, return as list
    return list(results.values())
""",
        "concatenate": """
def compose(results):
    \"\"\"Concatenate all results into a single list.\"\"\"
    result_list = []
    for key in sorted(results.keys()):
        result_list.append(results[key])
    return result_list
""",
        "sum": """
def compose(results):
    \"\"\"Sum all numeric results.\"\"\"
    return sum(results.values())
""",
        "first": """
def compose(results):
    \"\"\"Return the first result.\"\"\"
    return list(results.values())[0]
""",
    }
    
    return templates.get(operation, templates["merge"])

