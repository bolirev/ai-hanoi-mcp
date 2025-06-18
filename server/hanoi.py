import logging

import mcp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Hanoi")


@mcp.tool()
def hanoi_solver(n: int) -> list[tuple[int, int, int]]:
    """Solve the Tower of Hanoi puzzle"""
    moves = []
    pegs = [list(range(n, 0, -1)), [], []]  # All disks on the first peg
    hanoi(n, moves, pegs)  # Generate the move sequence
    return moves


@mcp.tool()
def hanoi(n, moves, pegs, start_peg=0, auxiliary_peg=1, target_peg=2):
    """Solve the Tower of Hanoi puzzle"""
    if n == 1:
        # Base case: Move the smallest disk from the start peg to the target peg
        disk = pegs[start_peg][-1]
        moves.append((disk, start_peg, target_peg))  # Keep 0-based indexing
        # Perform the move
        pegs[target_peg].append(pegs[start_peg].pop())
        return

    # Recursive case
    # Move n-1 disks from start_peg to auxiliary_peg using target_peg
    hanoi(
        n - 1,
        moves,
        pegs,
        start_peg,
        auxiliary_peg=target_peg,
        target_peg=auxiliary_peg,
    )

    # Move the nth disk from start_peg to target_peg
    disk = pegs[start_peg][-1]
    moves.append((disk, start_peg, target_peg))  # Keep 0-based indexing
    pegs[target_peg].append(pegs[start_peg].pop())

    # Move n-1 disks from auxiliary_peg to target_peg using start_peg
    hanoi(
        n - 1,
        moves,
        pegs,
        start_peg=auxiliary_peg,
        target_peg=target_peg,
        auxiliary_peg=start_peg,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Hanoi MCP server")
    mcp.run(transport="stdio")
