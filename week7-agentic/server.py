from mcp.server.fastmcp import FastMCP

mcp = FastMCP("day8-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake = {"Tokyo": "22C sunny", "Osaka": "25C cloudy"}
    return fake.get(city, f"No data for {city}")

if __name__ == "__main__":
    mcp.run()