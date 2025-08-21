# tool_discovery.py

def format_tool_info(tool) -> str:
    """Return a formatted string describing a tool's usage."""
    name = tool.name
    description = tool.description or "No description available."

    # Handle parameter introspection
    param_str = ""
    if hasattr(tool, 'input_schema') and tool.input_schema and 'properties' in tool.input_schema:
        props = tool.input_schema['properties']
        param_str = ", ".join([f"{key}: {val.get('type', 'any')}" for key, val in props.items()])
    elif hasattr(tool, 'parameters') and tool.parameters:
        param_str = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters if hasattr(p, 'name') and hasattr(p, 'type')])

    if param_str:
        return f"🔧 **{name}({param_str})**\n➡️  {description}"
    else:
        return f"🔧 **{name}**\n➡️  {description}"

def generate_tool_help(tools: list) -> str:
    """Create a user-friendly help message for all available tools."""
    if not tools:
        return "⚠️ No tools are currently available."

    intro = "🛠️ I can use the following tools to help you:\n"
    tool_details = "\n\n".join([format_tool_info(tool) for tool in tools])

    closing = (
        "\n\n💡 You can ask me things like:\n"
        "- 'Insert a customer **add your details here** '\n"
        "- 'Delete row where impact is D'\n"
        "- 'How many rows are there?'\n"
        "- 'What's the schema of this table?'\n"
        "- 'Create a randm data entry'\n"
    )

    return intro + tool_details + closing
