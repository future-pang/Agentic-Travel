from typing import List
from langchain_core.messages import BaseMessage, SystemMessage
from server.agent.session_storage import TranscriptMessage
from utils.tokens import token_count_with_estimation
from utils.logger import get_logger

logger = get_logger("shiliu.session_compression")

MAX_CONTEXT_TOKENS = 200_000

def compress_messages(messages: List[TranscriptMessage]) -> List[TranscriptMessage]:
    """
    使用混合计费策略计算 Token，并在超过 MAX_CONTEXT_TOKENS (200K) 时，
    从前往后丢弃最旧的非 system 消息，直至满足大小限制。
    实时重新评估以避免累积误差。
    """
    current_tokens = token_count_with_estimation(messages)
    if current_tokens <= MAX_CONTEXT_TOKENS:
        return messages

    system_msgs = [m for m in messages if m.role == "system" or m.type == "system"]
    non_sys_msgs = [m for m in messages if m.role != "system" and m.type != "system"]

    tokens_to_drop = current_tokens - MAX_CONTEXT_TOKENS
    dropped_count = 0

    while non_sys_msgs and tokens_to_drop > 0:
        non_sys_msgs.pop(0)
        dropped_count += 1
        
        # 实时重新评估剩余总量
        remaining = system_msgs + non_sys_msgs
        current_tokens = token_count_with_estimation(remaining, original_messages=messages)
        if current_tokens <= MAX_CONTEXT_TOKENS:
            break
        tokens_to_drop = current_tokens - MAX_CONTEXT_TOKENS

    kept_uuids = {m.uuid for m in system_msgs + non_sys_msgs}
    return [m for m in messages if m.uuid in kept_uuids]

def trim_context(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    对实时对话中的 LangChain BaseMessage 列表进行上下文窗口裁剪。
    使用混合计费策略，超过 MAX_CONTEXT_TOKENS 时从前往后丢弃最旧的非 system 消息。
    """
    if not messages:
        return messages

    current_tokens = token_count_with_estimation(messages)
    if current_tokens <= MAX_CONTEXT_TOKENS:
        return messages

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_sys_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    tokens_to_drop = current_tokens - MAX_CONTEXT_TOKENS
    dropped_count = 0

    while non_sys_msgs and tokens_to_drop > 0:
        non_sys_msgs.pop(0)
        dropped_count += 1
        
        remaining = system_msgs + non_sys_msgs
        current_tokens = token_count_with_estimation(remaining, original_messages=messages)
        if current_tokens <= MAX_CONTEXT_TOKENS:
            break
        tokens_to_drop = current_tokens - MAX_CONTEXT_TOKENS

    kept_uuids = {m.id for m in system_msgs + non_sys_msgs}
    result = [m for m in messages if m.id in kept_uuids]

    if dropped_count > 0:
        logger.info(
            "上下文窗口压缩完成",
            extra={
                "original_count": len(messages),
                "dropped_count": dropped_count,
                "kept_count": len(result),
                "estimated_tokens": token_count_with_estimation(result),
            }
        )
    return result
