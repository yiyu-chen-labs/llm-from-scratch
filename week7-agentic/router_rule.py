import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).parent
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(HERE / "server.py")],
)


def route(query: str):
    """Rule-based routing. Cheap, predictable, brittle."""
    if "weather" in query.lower() or "天気" in query:
        city = "Tokyo" if "tokyo" in query.lower() else "Osaka"
        return "get_weather", {"city": city}
    if "+" in query:
        a, b = query.split("+")
        return "add", {"a": int(a.strip()), "b": int(b.strip())}
    return None, None


async def main():
    queries = [
        "5 + 3",
        "what's the weather in Tokyo?",
        "東京の天気は？",
        "How much is five plus three?",   # 故意讓規則路由失敗
    ]

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            available = {t.name for t in (await session.list_tools()).tools}

            for q in queries:
                name, args = route(q)
                if name is None:
                    print(f"{q!r:45} -> NO ROUTE")
                    continue
                if name not in available:
                    print(f"{q!r:45} -> UNKNOWN TOOL {name}")
                    continue
                res = await session.call_tool(name, arguments=args)
                res = await session.call_tool(name, arguments=args)
                # print(res)                    # 看完整物件
                # print(res.content)            # 看 content 那層
                # print(res.structuredContent)  # 看結構化那層
                print(f"{q!r:45} -> {name}{args} = {res.content[0].text}")


if __name__ == "__main__":
    asyncio.run(main())