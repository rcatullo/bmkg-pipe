# Architecture Documentation

This document provides an in-depth look at the Semantic Parser Framework architecture, design decisions, and extensibility points.

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ReACT Engine                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Reasoning Loop:                                           │  │
│  │  1. Build Prompt (PromptBuilder)                         │  │
│  │  2. Query LLM (LLMClient)                                │  │
│  │  3. Parse Response                                        │  │
│  │  4. Execute Action (ActionRegistry)                      │  │
│  │  5. Record Step (ReasoningTrace)                         │  │
│  │  6. Repeat or Finish                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
              │              │              │              │
              ▼              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
       │   LLM    │   │  Action  │   │  Prompt  │   │Reasoning │
       │  Client  │   │ Registry │   │ Builder  │   │  Trace   │
       └──────────┘   └──────────┘   └──────────┘   └──────────┘
              │              │
              ▼              ▼
       ┌──────────┐   ┌──────────┐
       │  Azure   │   │  Custom  │
       │ OpenAI   │   │ Actions  │
       │    o3    │   │          │
       └──────────┘   └──────────┘
```

## Core Components

### 1. ReACT Engine (`react_engine.py`)

**Purpose**: Orchestrates the reasoning loop and manages the overall parsing process.

**Key Responsibilities**:
- Initialize and manage reasoning traces
- Coordinate between LLM and actions
- Handle step execution and observation recording
- Provide streaming and batch interfaces

**Design Decisions**:
- **Separation of Concerns**: Engine focuses on orchestration, not implementation
- **Configurability**: Max steps and verbosity are configurable
- **Error Resilience**: Continues operation even if individual actions fail

**Extension Points**:
```python
class CustomReACTEngine(ReACTEngine):
    def _get_next_action(self, trace, step_num):
        # Override to add custom logic before/after LLM call
        # Example: Add caching, retry logic, or custom prompt manipulation
        pass
    
    def _execute_action(self, action_call):
        # Override to add middleware, logging, or validation
        pass
```

### 2. Action Protocol (`action_protocol.py`)

**Purpose**: Defines the contract that all actions must follow.

**Design Pattern**: Template Method Pattern
- Abstract base class defines the interface
- Concrete implementations provide specific behavior
- Framework code interacts only with the interface

**Key Classes**:

#### `Action` (Abstract Base Class)
```python
class Action(ABC):
    # Template method
    def get_action_spec(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_input_schema()  # Hook method
        }
    
    @abstractmethod
    def execute(self, **kwargs) -> ActionOutput:  # Hook method
        pass
```

#### `ActionRegistry` (Registry Pattern)
- Manages all available actions
- Provides discovery mechanism for the LLM
- Enables dynamic action registration/unregistration

**Why This Design**:
- **Extensibility**: Easy to add new actions without modifying framework
- **Type Safety**: Pydantic models ensure correct data structures
- **Discoverability**: Registry makes actions available to LLM automatically

### 3. LLM Client (`llm_client.py`)

**Purpose**: Abstract away LLM API details and provide a clean interface.

**Design Pattern**: Adapter Pattern
- Adapts Azure OpenAI API to framework's needs
- Provides consistent interface regardless of underlying provider

**Key Features**:
- Environment variable configuration
- Streaming and non-streaming modes
- Response standardization via `LLMResponse` model

**Future Extensions**:
```python
class LLMClientFactory:
    @staticmethod
    def create(provider: str, **config):
        if provider == "azure":
            return AzureOpenAIClient(**config)
        elif provider == "openai":
            return OpenAIClient(**config)
        elif provider == "anthropic":
            return AnthropicClient(**config)
```

### 4. Reasoning Step (`reasoning_step.py`)

**Purpose**: Model and track the reasoning process.

**Design Pattern**: Value Object Pattern
- Immutable data structures representing reasoning state
- Rich domain model with behavior (not just data)

**Key Models**:

```
ReasoningTrace
├── query: str
├── target_format: str
├── steps: List[ReasoningStep]
├── final_output: Optional[str]
└── metadata: Dict

ReasoningStep
├── step_number: int
├── thought: Thought
├── action: ActionCall
└── observation: Optional[Observation]

Thought
└── content: str

ActionCall
├── action_name: str
└── parameters: Dict

Observation
├── action_name: str
├── result: Any
├── success: bool
└── error: Optional[str]
```

**Benefits**:
- Complete audit trail
- Reproducibility
- Debugging capability
- Performance analysis

### 5. Prompt Builder (`prompt_builder.py`)

**Purpose**: Construct prompts that guide the LLM's reasoning.

**Design Considerations**:
- **Clarity**: Prompts must be unambiguous
- **Completeness**: Include all necessary context
- **Parsability**: LLM responses must be easy to parse

**Prompt Structure**:
```
System Prompt
├── ReACT instructions
├── Output format specification
└── Guidelines and constraints

User Prompt
├── Task description (query + target format)
├── Available actions with schemas
├── Reasoning history
├── Step counter and warnings
└── Next step prompt
```

**Extension Points**:
```python
class DomainSpecificPromptBuilder(PromptBuilder):
    def build_system_prompt(self) -> str:
        # Add domain-specific instructions
        base = super().build_system_prompt()
        return base + self._get_domain_instructions()
    
    def build_reasoning_prompt(self, trace, max_steps) -> str:
        # Add domain-specific context
        prompt = super().build_reasoning_prompt(trace, max_steps)
        return self._inject_domain_context(prompt, trace)
```

## Data Flow

### Complete Parsing Flow

```
1. User Request
   └─> engine.parse(query, target_format)

2. Initialization
   └─> Create ReasoningTrace
   └─> Set initial context

3. For each step (up to max_steps):
   
   a. Prompt Construction
      └─> PromptBuilder.build_system_prompt()
      └─> PromptBuilder.build_reasoning_prompt(trace)
   
   b. LLM Reasoning
      └─> LLMClient.chat_completion(messages)
      └─> Parse response → (thought, action_name, parameters)
   
   c. Action Execution
      └─> ActionRegistry.get_action(action_name)
      └─> Action.validate_input(**parameters)
      └─> Action.execute(**parameters)
      └─> Return ActionOutput
   
   d. Observation Recording
      └─> Create Observation from ActionOutput
      └─> Create ReasoningStep
      └─> Add step to trace
   
   e. Completion Check
      └─> If action == "finish":
          └─> Extract final_output
          └─> trace.complete(final_output)
          └─> Break loop

4. Return
   └─> Return complete ReasoningTrace
```

### Streaming Flow

```
1. User Request
   └─> for step in engine.run_streaming(query, target_format):

2. For each step:
   └─> Yield ReasoningStep immediately after creation
   └─> User can process step in real-time
   └─> Continue to next iteration

3. On Completion:
   └─> Yield final ReasoningTrace
```

## Design Patterns Used

### 1. Template Method Pattern
- `Action` class defines template for all actions
- Subclasses implement specific behavior
- Framework calls template methods consistently

### 2. Registry Pattern
- `ActionRegistry` manages action instances
- Provides discovery and lookup
- Enables dynamic registration

### 3. Adapter Pattern
- `LLMClient` adapts Azure OpenAI API
- Provides consistent interface
- Isolates framework from API changes

### 4. Strategy Pattern
- Different actions = different strategies
- Selected dynamically by LLM
- Interchangeable implementations

### 5. Observer Pattern (Streaming)
- Engine yields events as they occur
- Observers (user code) react to events
- Decouples event production from consumption

### 6. Builder Pattern
- `PromptBuilder` constructs complex prompts
- Step-by-step construction
- Produces different prompt variations

## Extensibility Points

### 1. Custom Actions

**Location**: User-defined modules
**Interface**: `Action` base class
**Registration**: `ActionRegistry.register()`

```python
class MyAction(Action):
    def __init__(self):
        super().__init__(name="my_action", description="...")
    
    def get_input_schema(self) -> Dict:
        return {...}
    
    def execute(self, **kwargs) -> ActionOutput:
        return ActionOutput(...)
```

### 2. Custom Prompt Strategies

**Location**: Subclass `PromptBuilder`
**Override**: `build_system_prompt()`, `build_reasoning_prompt()`

```python
class SQLPromptBuilder(PromptBuilder):
    def build_system_prompt(self) -> str:
        return super().build_system_prompt() + SQL_SPECIFIC_INSTRUCTIONS
```

### 3. Custom LLM Clients

**Location**: Implement same interface as `AzureOpenAIClient`
**Interface**: `chat_completion()` method returning `LLMResponse`

```python
class CustomLLMClient:
    def chat_completion(self, messages: List[Message], **kwargs) -> LLMResponse:
        # Custom implementation
        pass
```

### 4. Custom Engine Behavior

**Location**: Subclass `ReACTEngine`
**Override**: `_get_next_action()`, `_execute_action()`

```python
class CachingReACTEngine(ReACTEngine):
    def __init__(self, *args, cache, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = cache
    
    def _execute_action(self, action_call):
        cache_key = (action_call.action_name, frozenset(action_call.parameters.items()))
        if cache_key in self.cache:
            return self.cache[cache_key]
        result = super()._execute_action(action_call)
        self.cache[cache_key] = result
        return result
```

### 5. Middleware System

Could be added for action execution:

```python
class ActionMiddleware(ABC):
    @abstractmethod
    def before_execute(self, action_call: ActionCall) -> ActionCall:
        pass
    
    @abstractmethod
    def after_execute(self, result: ActionOutput) -> ActionOutput:
        pass

class ReACTEngineWithMiddleware(ReACTEngine):
    def __init__(self, *args, middleware: List[ActionMiddleware], **kwargs):
        super().__init__(*args, **kwargs)
        self.middleware = middleware
    
    def _execute_action(self, action_call):
        # Before middleware
        for mw in self.middleware:
            action_call = mw.before_execute(action_call)
        
        # Execute
        result = super()._execute_action(action_call)
        
        # After middleware
        for mw in reversed(self.middleware):
            result = mw.after_execute(result)
        
        return result
```

## Error Handling Strategy

### Levels of Error Handling

1. **Action Level**: Actions catch and return errors in `ActionOutput`
2. **Engine Level**: Engine handles action failures gracefully
3. **Application Level**: User code handles engine failures

### Error Flow

```
Action Fails
└─> Returns ActionOutput(success=False, error="...")
    └─> Observation created with error
        └─> Added to reasoning trace
            └─> LLM sees error in next step
                └─> Can adjust strategy
```

**Benefits**:
- Errors don't crash the system
- LLM can recover from failures
- Complete error history in trace

## Performance Considerations

### Optimization Strategies

1. **Action Caching**
   - Cache expensive operations (schema retrieval, API calls)
   - Invalidate cache appropriately

2. **Prompt Optimization**
   - Limit history included in prompts
   - Summarize old steps

3. **Parallel Execution**
   - Execute independent actions in parallel
   - Requires modification to engine

4. **Streaming**
   - Use streaming for long-running tasks
   - Provides incremental feedback

### Scalability

The framework is designed for:
- **Vertical Scaling**: Single-threaded per query
- **Horizontal Scaling**: Process multiple queries in parallel

For high throughput:
```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)
futures = [executor.submit(engine.parse, query, format) for query in queries]
results = [f.result() for f in futures]
```

## Security Considerations

### Input Validation
- All action inputs validated against schema
- Prevents injection attacks

### LLM Output Parsing
- Robust parsing handles malformed responses
- Prevents code execution from LLM output

### Action Isolation
- Actions should validate their inputs
- Should not trust LLM output blindly

### Best Practices
```python
def execute(self, query: str, **kwargs):
    # Validate and sanitize
    if not self._is_safe_query(query):
        return ActionOutput(success=False, error="Unsafe query")
    
    # Use parameterized queries
    result = self.db.execute_safe(query, parameters)
    
    return ActionOutput(success=True, result=result)
```

## Testing Strategy

### Unit Tests
- Test individual actions
- Test prompt building
- Test response parsing

### Integration Tests
- Test engine with mock LLM
- Test action registry
- Test complete flows

### End-to-End Tests
- Test with real LLM (expensive)
- Verify complete parsing flows
- Test error recovery

## Future Enhancements

### Possible Additions

1. **Multi-Agent System**
   - Specialized agents for different tasks
   - Agent coordination protocol

2. **Learning System**
   - Learn from successful parses
   - Improve action selection over time

3. **Parallel Reasoning**
   - Explore multiple paths simultaneously
   - Select best result

4. **Interactive Mode**
   - User can intervene during reasoning
   - Provide hints or corrections

5. **Plugin System**
   - Hot-reload actions
   - Community action marketplace

## Conclusion

The framework is designed with:
- **Modularity**: Components are independent
- **Extensibility**: Easy to add new capabilities
- **Maintainability**: Clear separation of concerns
- **Debuggability**: Complete reasoning traces

These design principles ensure the framework can evolve with your needs while maintaining stability and performance.
