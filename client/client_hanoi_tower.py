# Create server parameters for stdio connection
import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config.hanoi_config import HanoiConfig, HanoiSolution

load_dotenv()


async def run_agent(config: HanoiConfig = HanoiConfig()):
    """
    Run the Tower of Hanoi solving agent.

    Returns:
        dict: Agent response containing messages and structured solution
    """

    # Create model using configuration
    model = ChatOpenAI(
        model=config.model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create server parameters using configuration
    server_params = StdioServerParameters(
        command=config.server_command,
        args=config.server_args,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Create and run the agent
            agent = create_react_agent(
                model,
                tools=await load_mcp_tools(session) if config.use_mcp else [],
                prompt=config.build_system_prompt(),
                response_format=HanoiSolution,
            )

            agent_response = await agent.ainvoke(
                {"messages": [config.get_puzzle_message()]}
            )
            return agent_response


def main():
    """Main function with command-line argument parsing."""
    # Default values
    default_n_disks = 2

    result = asyncio.run(run_agent(HanoiConfig(n_disks=default_n_disks)))

    print("Full response:")
    print(result)

    # Extract the structured response
    if "structured_response" in result:
        print("\nStructured solution:")
        solution = result["structured_response"]
        print(f"Total moves: {solution.total_moves}")
        print("Moves:")
        for i, move in enumerate(solution.moves, 1):
            print(
                f"  {i}. Move disk {move.disk_id} from peg {move.from_peg} to peg {move.to_peg}"
            )
    else:
        print("\nNo structured response found in the result.")


# Run the async function
if __name__ == "__main__":
    main()
