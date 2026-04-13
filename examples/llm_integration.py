"""LLM integration demo: OpenAI function calling tools for TetraMem."""

from tetrahedron_memory import GeoMemoryBody
from tetrahedron_memory.llm_tool import (
    create_tool_response,
    execute_tool_call,
    get_tool_definitions,
)


def main():
    # 1. Initialize memory
    memory = GeoMemoryBody(dimension=3, precision="fast")

    # 2. Show available tool definitions (pass these to OpenAI API)
    tools = get_tool_definitions()
    print(f"Available tools ({len(tools)}):")
    for t in tools:
        func = t["function"]
        print(f"  {func['name']}: {func['description'][:60]}...")

    # 3. Simulate what an LLM would do: call tools to manage memory
    #    In production, OpenAI returns tool_calls in the chat response

    # Store a memory via tool call
    result = execute_tool_call(
        memory,
        tool_name="tetramem_store",
        arguments={
            "content": "Persistent Homology reveals topological invariants",
            "labels": ["topology", "ph"],
            "weight": 1.0,
        },
    )
    print(f"\nStore result: {result}")

    # Store another
    result = execute_tool_call(
        memory,
        tool_name="tetramem_store",
        arguments={
            "content": "Alpha Complex triangulates weighted point sets",
            "labels": ["topology", "alpha"],
            "weight": 1.1,
        },
    )
    print(f"Store result: {result}")

    # Query
    result = execute_tool_call(
        memory,
        tool_name="tetramem_query",
        arguments={"query_text": "topology", "k": 3},
    )
    print(f"\nQuery result: {result}")

    # Associate
    node_ids = list(memory._nodes.keys())
    if node_ids:
        result = execute_tool_call(
            memory,
            tool_name="tetramem_associate",
            arguments={"memory_id": node_ids[0], "max_depth": 2},
        )
        print(f"\nAssociate result: {result}")

    # Stats
    result = execute_tool_call(memory, tool_name="tetramem_stats", arguments={})
    print(f"\nStats: {result}")

    # 4. Show how to create a tool response for OpenAI
    response = create_tool_response("call_abc123", result)
    print(
        f"\nTool response for OpenAI: role={response['role']}, tool_call_id={response['tool_call_id']}"
    )


if __name__ == "__main__":
    main()
