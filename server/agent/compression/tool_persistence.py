import os
import json
from typing import Any, Dict
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from utils.logger import get_logger

# 触发条件：工具结果字符数超过 50,000
DEFAULT_MAX_RESULT_SIZE_CHARS = 50_000

logger = get_logger("shiliu.tool_persistence")

class PersistingToolNode(ToolNode):
    """
    自定义的 ToolNode，会将过大的工具结果持久化到磁盘。
    如果 ToolMessage 的内容超过阈值，会将其保存为文件，
    并将内容替换为包含预览的 JSON 元数据，以节省上下文空间。
    """

    async def ainvoke(self, input: Any, config: Dict[str, Any], **kwargs: Any) -> Any:
        result = await super().ainvoke(input, config, **kwargs)
        if not isinstance(result, dict) or "messages" not in result:
            return result

        messages = result["messages"]
        session_id = config.get("configurable", {}).get("thread_id", "default_session")

        history_dir = os.path.join(".data", "chat_history", session_id, "tool-results")
        os.makedirs(history_dir, exist_ok=True)

        processed_messages = []

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                processed_messages.append(msg)
                continue

            # 提取多模态内容中的纯文本
            def extract_text(c):
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    texts = []
                    for block in c:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    return "\n".join(texts)
                return str(c)
                
            content_str = extract_text(msg.content)
            original_size = len(content_str)

            if original_size > DEFAULT_MAX_RESULT_SIZE_CHARS:
                tool_call_id = msg.tool_call_id

                # 保存为 .txt，因为工具返回值可能是字符串或 JSON
                file_name = f"{tool_call_id}.txt"
                file_path = os.path.join(history_dir, file_name)

                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content_str)
                except Exception as e:
                    logger.error(f"Failed to persist tool result: {e}")
                    processed_messages.append(msg)
                    continue

                preview = content_str[:2048]

                replacement_dict = {
                    "persisted_output": True,
                    "original_size_bytes": len(content_str.encode("utf-8")),
                    "file_path": file_path,
                    "preview": preview + ("..." if original_size > 2048 else "")
                }

                msg.content = json.dumps(replacement_dict, ensure_ascii=False)

            processed_messages.append(msg)

        result["messages"] = processed_messages
        return result