# ReACT Engine Frontend

A beautiful web interface for visualizing the reasoning trace of the ReACT semantic parsing engine.

![ReACT Engine UI](https://via.placeholder.com/800x400?text=ReACT+Engine+UI)

## Features

- 🧠 **Real-time Reasoning Trace** - See each step of the ReACT loop as it happens
- 💭 **Thought Visualization** - View the LLM's reasoning process
- ⚡ **Action Tracking** - Monitor which actions are executed and their parameters
- 📊 **Observation Display** - See the results from each action execution
- ✅ **Final Output** - Get the complete answer with execution statistics

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Set Environment Variables

Make sure your `.env` file in the project root contains:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Server

```bash
# From the frontend directory
python server.py

# Or using uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open in Browser

Navigate to [http://localhost:8000](http://localhost:8000)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the main UI |
| `/api/parse` | POST | Parse a query and return the full trace |
| `/api/parse/stream` | POST | Parse with Server-Sent Events streaming |
| `/api/health` | GET | Health check |
| `/api/module-info` | GET | Get loaded module information |

## Usage

1. Enter your natural language query in the text area
2. Adjust the maximum steps slider if needed
3. Click "Run Query"
4. Watch the reasoning trace appear step by step

## Example Queries (Verdant Module)

- "What is the average MOIC for all funds in 2024?"
- "Show the top 5 funds by total invested capital"
- "Compare Q4 2024 vs Q4 2023 MOIC for all funds"
- "Which funds have MOIC greater than 2.0?"

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Browser UI     │────▶│   FastAPI        │────▶│  ReACT Engine    │
│   (index.html)   │◀────│   (server.py)    │◀────│  (semantic_parser)│
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

## Customization

### Changing the Module

Edit `server.py` to use a different module:

```python
# For Cypher (Neo4j)
from semantic_parser.modules.cypher import create_action_registry

# For Spider2
from semantic_parser.modules.spider2 import create_action_registry
```

### Styling

The UI uses CSS custom properties for theming. Edit the `:root` variables in `index.html`:

```css
:root {
    --bg-primary: #0a0a0f;
    --accent-blue: #5b8def;
    /* ... */
}
```

## Troubleshooting

### "Module not found" errors

Make sure you're running the server from the `frontend` directory or that the parent directory is in your Python path.

### "API key not found"

Ensure your `.env` file is in the project root and contains `AZURE_OPENAI_API_KEY`.

### CORS issues

The server includes CORS middleware that allows all origins by default. For production, restrict this to specific domains.




