import os
import sys
import asyncio
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import ToolMessage
from server.agent.compression.tool_persistence import PersistingToolNode

# We'll create a dummy tool function
def dummy_tool(query: str):
    """Dummy tool for testing tool result persistence."""
    return "A" * 60_000

async def main():
    print("=" * 60)
    print(" Testing Tool Result Persistence (50K chars limit)")
    print("=" * 60)
    
    session_id = "test_tool_001"
    
    from langgraph.prebuilt import ToolNode
    from unittest.mock import AsyncMock
    ToolNode.ainvoke = AsyncMock(return_value={
        "messages": [
            ToolMessage(
                content="A" * 60_000,
                tool_call_id="call_abc123",
                name="dummy_tool",
                id="dummy_msg_id"
            )
        ]
    })

    node = PersistingToolNode([dummy_tool])
    
    # Simulate an invocation where the LLM requested a tool
    input_state = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "dummy_tool", "args": {"query": "hello"}, "id": "call_abc123"}]
            }
        ]
    }
    config = {"configurable": {"thread_id": session_id}}
    
    # Execute the node
    result = await node.ainvoke(input_state, config)
    
    messages = result.get("messages", [])
    print(f"Returned messages count: {len(messages)}")
    assert len(messages) == 1
    
    tool_msg = messages[0]
    assert isinstance(tool_msg, ToolMessage)
    
    # Check if content was persisted
    content_dict = json.loads(tool_msg.content)
    assert content_dict.get("persisted_output") is True
    assert content_dict.get("original_size_bytes") >= 60_000
    
    file_path = content_dict.get("file_path")
    assert os.path.exists(file_path), "Persisted file was not found!"
    
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
        assert len(file_content) == 60_000
        assert file_content.startswith("AAA")
        
    preview = content_dict.get("preview")
    assert len(preview) > 2000 # 2048 + "..."
    assert preview.startswith("AAA")
    
    # Clean up
    if os.path.exists(file_path):
        os.remove(file_path)
    
    print("\n[SUCCESS] Tool persistence test passed!")

if __name__ == "__main__":
    asyncio.run(main())
