import asyncio
import sys
from pathlib import Path

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).parent
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(HERE / "server.py")],
)

llm = anthropic.Anthropic()


async def main():
    queries = [
        "5 + 3",
        "How much is five plus three?",
        "東京の天気は？",
    ]

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            available = {t.name for t in mcp_tools}

            # MCP schema -> Anthropic tool format
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in mcp_tools
            ]

            for q in queries:
                msg = llm.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    tools=tools,
                    messages=[{"role": "user", "content": q}],
                )

                calls = [b for b in msg.content if b.type == "tool_use"]
                if not calls:
                    text = "".join(b.text for b in msg.content if b.type == "text")
                    print(f"{q!r:35} -> NO TOOL: {text[:60]}")
                    continue

                call = calls[0]
                if call.name not in available:
                    print(f"{q!r:35} -> HALLUCINATED {call.name}")
                    continue

                res = await session.call_tool(call.name, arguments=call.input)
                print(f"{q!r:35} -> {call.name}{call.input} = {res.content[0].text}")


if __name__ == "__main__":
    asyncio.run(main())