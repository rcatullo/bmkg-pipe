"""
FastAPI Server for ReACT Engine Frontend

This server exposes the ReACT engine as an API and serves the frontend.
Supports both ReACT and Baseline engines with comparison capabilities.
"""

import os
import sys
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_parser import ReACTEngine, AzureOpenAIClient
from semantic_parser.modules.verdant import create_action_registry
from semantic_parser.src.baseline import BaselineEngine

# Load environment variables
load_dotenv()

EXCEL_FILE_DIR = "/home/jiuding/computational_thinking/semantic_parser/src/baseline/db.xlsx"

app = FastAPI(
    title="Semantic Parser UI",
    description="Interactive frontend for ReACT and Baseline semantic parsing engines",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    max_steps: int = 10
    model: str = "o3"  # "gpt-5", "o3", "gpt-4o", etc.
    engine: str = "react"  # "react" or "baseline"


class StepResponse(BaseModel):
    step_number: int
    thought: str
    action_name: str
    action_params: Dict[str, Any]
    observation: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: str


class TraceResponse(BaseModel):
    query: str
    target_format: str
    steps: List[StepResponse]
    final_output: Optional[str] = None
    is_complete: bool
    duration_seconds: Optional[float] = None
    engine_type: str = "react"


class BaselineResponse(BaseModel):
    query: str
    answer: str
    tables_used: List[str]
    total_rows: int
    duration_seconds: float
    engine_type: str = "baseline"


class CompareRequest(BaseModel):
    query: str
    max_steps: int = 10
    model: str = "o3"


class CompareResponse(BaseModel):
    query: str
    react_result: Optional[TraceResponse] = None
    baseline_result: Optional[BaselineResponse] = None
    react_error: Optional[str] = None
    baseline_error: Optional[str] = None


# Global engine caches
_react_engines: Dict[str, ReACTEngine] = {}
_baseline_engines: Dict[str, BaselineEngine] = {}


def get_react_engine(model: str = "o3") -> ReACTEngine:
    """Get or create the ReACT engine for the specified model."""
    global _react_engines
    
    if model not in _react_engines:
        llm_client = AzureOpenAIClient(
            azure_endpoint="https://ovalnairr.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview",
            deployment_name=model,
        )
        
        # Create action registry with the same model for actions that use LLM
        action_registry = create_action_registry(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=model,
        )
        
        _react_engines[model] = ReACTEngine(
            llm_client=llm_client,
            action_registry=action_registry,
            max_steps=10,
            verbose=True
        )
    
    return _react_engines[model]


def get_baseline_engine(model: str = "o3") -> BaselineEngine:
    """Get or create the Baseline engine for the specified model."""
    global _baseline_engines
    
    if model not in _baseline_engines:
        llm_client = AzureOpenAIClient(
            azure_endpoint="https://ovalnairr.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview",
            deployment_name=model,
        )
        
        _baseline_engines[model] = BaselineEngine(
            llm_client=llm_client,
            xlsx_paths=None,  # Will use default path
            max_rows_per_table=500,
            smart_load=True,
            verbose=True
        )
    
    return _baseline_engines[model]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.get("/compare", response_class=HTMLResponse)
async def compare_page():
    """Serve the comparison page."""
    html_path = Path(__file__).parent / "compare.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Compare page not found</h1>", status_code=404)


@app.post("/api/baseline")
async def baseline_query(request: QueryRequest) -> BaselineResponse:
    """
    Run a query through the baseline engine.
    """
    try:
        engine = get_baseline_engine(model=request.model)
        engine.load_xlsx([EXCEL_FILE_DIR])
        # Run the query
        result = engine.query(query=request.query)
        
        return BaselineResponse(
            query=result.query,
            answer=result.answer,
            tables_used=result.tables_used,
            total_rows=result.total_rows,
            duration_seconds=result.duration_seconds,
            engine_type="baseline"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/parse")
async def parse_query(request: QueryRequest):
    """
    Parse a natural language query using the specified engine.
    """
    # Route to baseline if requested
    if request.engine == "baseline":
        return await baseline_query(request)
    
    # Otherwise use ReACT engine
    try:
        engine = get_react_engine(model=request.model)
        engine.max_steps = request.max_steps
        
        # Run the parsing
        trace = engine.parse(query=request.query)
        
        # Convert to response format
        steps = []
        for step in trace.steps:
            obs_result = None
            obs_success = True
            obs_error = None
            
            if step.observation:
                obs_result = str(step.observation.result) if step.observation.result else None
                obs_success = step.observation.success
                obs_error = step.observation.error
            
            steps.append(StepResponse(
                step_number=step.step_number,
                thought=step.thought.content,
                action_name=step.action.action_name,
                action_params=step.action.parameters,
                observation=obs_result,
                success=obs_success,
                error=obs_error,
                timestamp=step.thought.timestamp.isoformat()
            ))
        
        return TraceResponse(
            query=trace.query,
            target_format=trace.target_format,
            steps=steps,
            final_output=trace.final_output,
            is_complete=trace.is_complete,
            duration_seconds=trace.get_duration(),
            engine_type="react"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
async def compare_engines(request: CompareRequest) -> CompareResponse:
    """
    Run a query through both engines and return results side by side.
    """
    react_result = None
    baseline_result = None
    react_error = None
    baseline_error = None
    
    # Run ReACT engine
    try:
        engine = get_react_engine(model=request.model)
        engine.max_steps = request.max_steps
        trace = engine.parse(query=request.query)
        
        steps = []
        for step in trace.steps:
            obs_result = None
            obs_success = True
            obs_error = None
            
            if step.observation:
                obs_result = str(step.observation.result) if step.observation.result else None
                obs_success = step.observation.success
                obs_error = step.observation.error
            
            steps.append(StepResponse(
                step_number=step.step_number,
                thought=step.thought.content,
                action_name=step.action.action_name,
                action_params=step.action.parameters,
                observation=obs_result,
                success=obs_success,
                error=obs_error,
                timestamp=step.thought.timestamp.isoformat()
            ))
        
        react_result = TraceResponse(
            query=trace.query,
            target_format=trace.target_format,
            steps=steps,
            final_output=trace.final_output,
            is_complete=trace.is_complete,
            duration_seconds=trace.get_duration(),
            engine_type="react"
        )
    except Exception as e:
        react_error = str(e)
    
    # Run Baseline engine
    try:
        baseline_eng = get_baseline_engine(model=request.model)
        baseline_eng.load_xlsx([EXCEL_FILE_DIR])
        result = baseline_eng.query(query=request.query)
        
        baseline_result = BaselineResponse(
            query=result.query,
            answer=result.answer,
            tables_used=result.tables_used,
            total_rows=result.total_rows,
            duration_seconds=result.duration_seconds,
            engine_type="baseline"
        )
    except Exception as e:
        baseline_error = str(e)
    
    return CompareResponse(
        query=request.query,
        react_result=react_result,
        baseline_result=baseline_result,
        react_error=react_error,
        baseline_error=baseline_error
    )


@app.post("/api/parse/stream")
async def parse_query_stream(request: QueryRequest):
    """
    Parse a query and stream the reasoning steps as they happen.
    """
    async def generate():
        try:
            engine = get_react_engine(model=request.model)
            engine.max_steps = request.max_steps
            
            # Run parsing in a thread pool
            loop = asyncio.get_event_loop()
            trace = await loop.run_in_executor(
                None,
                lambda: engine.parse(query=request.query)
            )
            
            # Stream each step
            for step in trace.steps:
                obs_result = None
                obs_success = True
                obs_error = None
                
                if step.observation:
                    obs_result = str(step.observation.result) if step.observation.result else None
                    obs_success = step.observation.success
                    obs_error = step.observation.error
                
                step_data = {
                    "type": "step",
                    "step_number": step.step_number,
                    "thought": step.thought.content,
                    "action_name": step.action.action_name,
                    "action_params": step.action.parameters,
                    "observation": obs_result,
                    "success": obs_success,
                    "error": obs_error,
                    "timestamp": step.thought.timestamp.isoformat()
                }
                yield f"data: {json.dumps(step_data)}\n\n"
            
            # Send final result
            final_data = {
                "type": "complete",
                "final_output": trace.final_output,
                "is_complete": trace.is_complete,
                "duration_seconds": trace.get_duration()
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/module-info")
async def module_info():
    """Get information about the loaded module."""
    try:
        engine = get_react_engine()
        registry = engine.action_registry
        
        return {
            "module_name": registry.module_config.name if registry.module_config else "unknown",
            "target_format": registry.target_format,
            "predecided_actions": registry.predecided_actions,
            "available_actions": [spec["name"] for spec in registry.get_action_specs()],
            "engines_available": ["react", "baseline"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="ReACT Engine Frontend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   ReACT Engine Frontend                      ║
╠══════════════════════════════════════════════════════════════╣
║  Server running at: http://{args.host}:{args.port}                    ║
║                                                              ║
║  To access from local machine via SSH tunnel:                ║
║  ssh -L {args.port}:localhost:{args.port} user@this-server              ║
║  Then open: http://localhost:{args.port}                          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

