"""
Hanoi-specific configuration and prompts.
Contains system prompts and algorithm descriptions for the Tower of Hanoi solver.
"""

from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class HanoiMove(BaseModel):
    """Represents a single move in the Tower of Hanoi puzzle."""

    disk_id: int = Field(description="The ID of the disk being moved")
    from_peg: int = Field(
        description="The peg the disk is being moved from (0-indexed)"
    )
    to_peg: int = Field(description="The peg the disk is being moved to (0-indexed)")


class HanoiSolution(BaseModel):
    """Represents the complete solution to the Tower of Hanoi puzzle."""

    moves: List[HanoiMove] = Field(description="List of moves to solve the puzzle")
    total_moves: int = Field(description="Total number of moves in the solution")

    def validate_solution(self, n_disks: int) -> dict:
        """
        Validate the complete solution by simulating the game.

        Args:
            n_disks (int): Number of disks in the puzzle

        Returns:
            dict: Validation result with success status and details
        """
        # Initialize the game state: all disks on peg 0
        pegs = [list(range(n_disks, 0, -1)), [], []]  # [largest...smallest], [], []

        validation_result = {
            "is_valid": True,
            "errors": [],
            "final_state": None,
            "moves_validated": 0,
        }

        # Validate each move
        for i, move in enumerate(self.moves):
            move_result = self._validate_single_move(pegs, move, i + 1)

            if not move_result["valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].append(move_result["error"])
                break

            # Apply the move
            disk = pegs[move.from_peg].pop()
            pegs[move.to_peg].append(disk)
            validation_result["moves_validated"] += 1

        validation_result["final_state"] = pegs

        # Check if puzzle is solved (all disks moved to peg 2)
        if validation_result["is_valid"]:
            goal_state = [[], [], list(range(n_disks, 0, -1))]
            if pegs != goal_state:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Puzzle not solved: Final state {pegs} != Goal state {goal_state}"
                )

        # Validate total_moves matches actual moves
        if self.total_moves != len(self.moves):
            validation_result["errors"].append(
                f"total_moves ({self.total_moves}) doesn't match actual moves ({len(self.moves)})"
            )

        return validation_result

    def _validate_single_move(
        self, pegs: List[List[int]], move: HanoiMove, move_number: int
    ) -> dict:
        """
        Validate a single move against current game state.

        Args:
            pegs: Current state of all pegs
            move: The move to validate
            move_number: Move number for error reporting

        Returns:
            dict: Validation result for this move
        """
        # Check peg indices are valid
        if not (0 <= move.from_peg <= 2 and 0 <= move.to_peg <= 2):
            return {
                "valid": False,
                "error": f"Move {move_number}: Invalid peg indices ({move.from_peg}, {move.to_peg})",
            }

        # Check source peg is not empty
        if not pegs[move.from_peg]:
            return {
                "valid": False,
                "error": f"Move {move_number}: Cannot move from empty peg {move.from_peg}",
            }

        # Check the disk being moved matches the expected disk_id
        top_disk = pegs[move.from_peg][-1]
        if top_disk != move.disk_id:
            return {
                "valid": False,
                "error": f"Move {move_number}: Expected to move disk {move.disk_id}, but top disk is {top_disk}",
            }

        # Check Hanoi rule: larger disk cannot be placed on smaller disk
        if pegs[move.to_peg] and pegs[move.to_peg][-1] < move.disk_id:
            return {
                "valid": False,
                "error": f"Move {move_number}: Cannot place disk {move.disk_id} on smaller disk {pegs[move.to_peg][-1]}",
            }

        return {"valid": True, "error": None}


class HanoiConfig(BaseModel):
    """
    Configuration model for the Tower of Hanoi solver.

    This model defines all configurable parameters for different solving approaches.
    """

    n_disks: int = Field(default=3, description="Number of disks in the puzzle")

    # Solver behavior
    use_mcp: bool = Field(
        default=True,
        description="Whether to use MCP (Model Context Protocol) tools for interactive solving",
    )

    add_pseudocode: bool = Field(
        default=False,
        description="Whether to include pseudocode algorithm guidance in the system prompt",
    )

    mcp_version: int = Field(
        default=1,
        description="The version of the MCP server to use",
    )

    # Model configuration
    model_name: str = Field(
        default="gpt-4o", description="OpenAI model name to use for the solver"
    )

    # MCP server configuration
    server_command: str = Field(
        default="python", description="Command to run the MCP server"
    )

    server_args: List[str] = Field(
        default=["server/hanoi.py"],
        description="Arguments to pass to the MCP server command",
    )

    def filter_mcp_tools(self, tools: List[StructuredTool]) -> List[StructuredTool]:
        """
        Filter the MCP tools based on the configuration.
        """
        if self.mcp_version == 1:
            return [tool for tool in tools if tool.name == "hanoi_solver"]
        elif self.mcp_version == 2:
            return [tool for tool in tools if tool.name == "hanoi_solver_dict_output"]
        else:
            raise ValueError(f"Invalid MCP version: {self.mcp_version}")

    def get_base_system_prompt(self) -> str:
        """
        Get the base system prompt for the Tower of Hanoi solver.

        Returns:
            str: Base system prompt with general instructions
        """
        return """You are a helpful assistant that solves the Tower of Hanoi puzzle. 

    There are three pegs and n disks of different sizes stacked on the first peg. The disks are
    numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
    1. Only one disk can be moved at a time.
    2. Each move consists of taking the upper disk from one stack and placing it on top of
    another stack.
    3. A larger disk may not be placed on top of a smaller disk.
    The goal is to move the entire stack to the third peg.

    Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1],
    [], []], and a solution might be:
    moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2],
    [1, 1, 0], [2, 1, 2], [1, 0, 2]]

    This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.

    Requirements:
    • When exploring potential solutions in your thinking process, always include the corresponding complete list of moves.
    • The positions are 0-indexed (the leftmost peg is 0).
    • Ensure your final answer includes the complete list of moves."""

    def get_pseudocode_prompt(self) -> str:
        """
        Get the pseudocode algorithm prompt for the Tower of Hanoi solver.

        Returns:
            str: Pseudocode algorithm description
        """
        return """

    Here is a pseudocode of recursive algorithm to solve the puzzle:

    ALGORITHM Solve(n, source, target, auxiliary, moves)
    // n = number of disks to move
    // source = starting peg (0, 1, or 2)
    // target = destination peg (0, 1, or 2)
    // auxiliary = the unused peg (0, 1, or 2)
    // moves = list to store the sequence of moves

    IF n equals 1 THEN
        // Get the top disk from source peg
        disk = the top disk on the source peg
        // Add the move to our list: [disk_id, source, target]
        ADD [disk, source, target] to moves
        RETURN
    END IF

    // Move n-1 disks from source to auxiliary peg
    Solve(n-1, source, auxiliary, target, moves)

    // Move the nth disk from source to target
    disk = the top disk on the source peg
    ADD [disk, source, target] to moves

    // Move n-1 disks from auxiliary to target
    Solve(n-1, auxiliary, target, source, moves)

    END ALGORITHM

    To solve the entire puzzle of moving n disks from peg 0 to peg 2:
    1. Initialize an empty list 'moves'
    2. Execute Solve(n, 0, 2, 1, moves)
    3. The 'moves' list will contain the complete solution"""

    def build_system_prompt(self) -> str:
        """
        Build the complete system prompt based on configuration.

        Args:
            add_pseudocode (bool): Whether to include pseudocode algorithm

        Returns:
            str: Complete system prompt
        """
        prompt = self.get_base_system_prompt()

        if self.add_pseudocode:
            prompt += self.get_pseudocode_prompt()
        if self.use_mcp:
            prompt += self.get_mcp_prompt()
        prompt += self.add_steps()
        return prompt

    def add_steps(self) -> str:
        """
        Add the step by step prompt to the system prompt.
        """
        steps = """
        Steps
        """
        next_step = 1
        if self.use_mcp:
            steps += f"""
            {next_step}. Use the MCP servers above to answer the user query, not every MCP server will be relevant for a given query so you can choose which ones to invoke.  
            """
            next_step += 1
        if self.add_pseudocode:
            steps += f"""
            {next_step}. Understand the relevant ALGORITHM, not every ALGORITHM will be relevant for a given query so you can choose which ones to invoke.
            """
            next_step += 1
        steps += f"""
            {next_step}. Answer the user query based on the information you have gathered.
        """
        next_step += 1
        return steps

    def get_mcp_prompt(self) -> str:
        """
        Get the MCP prompt for the Tower of Hanoi solver.

        Returns:
            str: MCP prompt
        """
        if self.mcp_version == 1:
            return """ Here is the description of the tools you can use to solve the puzzle:
            - hanoi_solver(n: int): Solve the Tower of Hanoi puzzle
                It returns the list of moves to solve the puzzle as a list of tuples (disk_id, from_peg, to_peg).
            """
        elif self.mcp_version == 2:
            return """ Here is the description of the tools you can use to solve the puzzle:
            - hanoi_solver_dict_output(n: int): Solve the Tower of Hanoi puzzle
                It returns the list of moves to solve the puzzle as a list of dictionaries with the following keys: move_nb, disk, from_peg, to_peg.
            """

    def get_puzzle_message(self) -> str:
        """
        Generate the puzzle description message for a given number of disks.

        Args:
            n_disks (int): Number of disks in the puzzle

        Returns:
            str: Formatted puzzle description
        """
        return f"""
    I have a puzzle with {self.n_disks} disks of different sizes with
    Initial configuration:
    • Peg 0: {self.n_disks} (bottom), ... 2, 1 (top)
    • Peg 1: (empty)
    • Peg 2: (empty)
    Goal configuration:
    • Peg 0: (empty)
    • Peg 1: (empty)
    • Peg 2: {self.n_disks} (bottom), ... 2, 1 (top)
    Rules:
    • Only one disk can be moved at a time.
    • Only the top disk from any stack can be moved.
    • A larger disk may not be placed on top of a smaller disk.
    Find the sequence of moves to transform the initial configuration into the goal configuration.
    """

    # Legacy function name for backward compatibility
    def prompt_algorithm(self) -> str:
        """Legacy function name - use get_pseudocode_prompt() instead."""
        return self.get_pseudocode_prompt()
