from __future__ import annotations

from ollama_agent_kit.tools import ToolDefinition


def build_tools() -> list[ToolDefinition]:
    def greet(arguments: dict[str, object]) -> str:
        name = str(arguments["name"])
        return f"Hello, {name}!"

    return [
        ToolDefinition(
            name="greet",
            description="Greet a person by name.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The person's name.",
                    }
                },
                "required": ["name"],
            },
            handler=greet,
        )
    ]
