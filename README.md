# Tower of Hanoi MCP Solver

A Model Context Protocol (MCP) based Tower of Hanoi puzzle solver using OpenAI models and Pydantic for configuration management.

## Overview

This project provides a simplified, efficient solution for solving Tower of Hanoi puzzles using:
- **MCP (Model Context Protocol)** for interactive solving
- **Pydantic models** for type-safe configuration
- **Solution validation** to ensure correct puzzle solutions

## Project Structure

```
mcp_test/
├── config/
│   ├── hanoi_config.py      # Configuration models and validation
│   └── __init__.py
├── client/
│   ├── client_hanoi_tower.py  # Main client implementation
│   └── client_math_server.py  # Math server client (example)
├── server/
│   ├── hanoi.py             # MCP server for Hanoi puzzle
│   └── math_server.py       # Math MCP server (example)
├── pyproject.toml          # Poetry configuration
└── README.md
```

## Core Components

### 1. Configuration Models (`config/hanoi_config.py`)

#### HanoiMove
Represents a single move in the Tower of Hanoi puzzle:
```python
class HanoiMove(BaseModel):
    disk_id: int        # ID of the disk being moved
    from_peg: int       # Source peg (0-indexed)
    to_peg: int         # Target peg (0-indexed)
```

#### HanoiSolution
Represents the complete solution with validation:
```python
class HanoiSolution(BaseModel):
    moves: List[HanoiMove]    # List of moves to solve the puzzle
    total_moves: int          # Total number of moves
    
    def validate_solution(self, n_disks: int) -> dict:
        # Validates the complete solution by simulating the game
```

#### HanoiConfig
Main configuration class with environment variable support:
```python
class HanoiConfig(BaseModel):
    n_disks: int = 3              # Number of disks in the puzzle
    use_mcp: bool = True          # Whether to use MCP tools
    add_pseudocode: bool = False  # Include algorithm guidance
    model_name: str = "gpt-4o"    # OpenAI model to use
    server_command: str = "python"        # MCP server command
    server_args: List[str] = ["server/hanoi.py"]  # Server arguments
```

### 2. Client Implementation (`client/client_hanoi_tower.py`)

The main client provides:
- **Async MCP connection** handling
- **Agent creation** with configurable tools and prompts
- **Solution processing** and display

```python
async def run_agent(config: HanoiConfig = HanoiConfig()):
    """Run the Tower of Hanoi solving agent."""
    # Creates OpenAI model, MCP connection, and agent
    # Returns structured solution as HanoiSolution object
```

## Setup

### 1. Environment Configuration

```
# OpenAI API Configuration (required)
OPENAI_API_KEY=your_api_key_here
```

### 2. Install Dependencies

Using Poetry:
```bash
poetry install
```

## Usage

### Command Line

Run the solver directly:
```bash
python client/client_hanoi_tower.py
```

This will solve a 2-disk puzzle by default and display:
- Complete solution with moves
- Move-by-move breakdown
- Validation results

### Usage

see `analyses.ipynb`

## Configuration Options

### Model Selection
- `gpt-4o`: Latest GPT-4 optimized model (default)
- `gpt-4o-mini`: Faster, more cost-effective option
- `gpt-4`: Standard GPT-4 model

### MCP Settings
- `use_mcp`: Enable/disable MCP tools (default: True)
- `add_pseudocode`: Include algorithm guidance (default: False)
- `server_command`: Command to run MCP server (default: "python")
- `server_args`: Arguments for MCP server (default: ["server/hanoi.py"])

### Puzzle Settings
- `n_disks`: Number of disks in puzzle (default: 3)
