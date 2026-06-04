"""
Layer 5: Auto-Compact (全量摘要兜底)

在 Context Collapse 没压住（或者主动停用）时，作为兜底的全量压缩策略。
触发点：167K tokens (针对 200K 模型)。
压缩后会执行 Post-Compact Context Restoration，恢复核心上下文（如最新的文件操作状态）。
"""
import uuid
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from utils.tokens import rough_estimation_for_messages
from utils.logger import get_logger

logger = get_logger("shiliu.auto_compact")

# ============================================================================
# 阈值与配置
# ============================================================================
AUTOCOMPACT_THRESHOLD = 167_000
MAX_CONSECUTIVE_FAILURES = 3

@dataclass
class AutoCompactResult:
    was_compacted: bool
    messages: List[BaseMessage]
    error: str = ""

# 熔断器状态
_consecutive_failures = 0


async def autocompact_if_needed(messages: List[BaseMessage]) -> AutoCompactResult:
    """全量摘要。只有在 context_collapse 没有把消息压下来时才触发。"""
    global _consecutive_failures
    
    token_count = rough_estimation_for_messages(messages)
    if token_count < AUTOCOMPACT_THRESHOLD:
        return AutoCompactResult(was_compacted=False, messages=messages)
        
    if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        logger.warning(f"Auto-Compact 处于熔断状态（连续失败 {_consecutive_failures} 次），跳过压缩。")
        return AutoCompactResult(was_compacted=False, messages=messages, error="Circuit broken")
        
    logger.info(f"触发 Layer 5: Auto-Compact 全量压缩 (当前 Tokens: {token_count} >= {AUTOCOMPACT_THRESHOLD})")
    
    try:
        # 保留最近的活跃对话 (例如最后 10 条，避免正在进行的思考被打断)
        KEEP_RECENT = 10
        if len(messages) <= KEEP_RECENT + 2:
            return AutoCompactResult(was_compacted=False, messages=messages)
            
        # 寻找要摘要的范围 (除 SystemPrompt 和最近消息之外的部分)
        system_msgs = [m for m in messages if isinstance(m, SystemMessage) and not m.additional_kwargs.get("is_collapsed_summary")]
        other_msgs = [m for m in messages if m not in system_msgs]
        
        if len(other_msgs) <= KEEP_RECENT:
            return AutoCompactResult(was_compacted=False, messages=messages)
            
        to_compact = other_msgs[:-KEEP_RECENT]
        recent_kept = other_msgs[-KEEP_RECENT:]
        
        # 1. 生成全局摘要
        summary_text = await _generate_global_summary(to_compact)
        
        # 2. 构造压缩后的历史 (带摘要)
        summary_msg = SystemMessage(
            content=f"【历史对话摘要】\n以下是之前的详细对话和工具调用已被压缩：\n{summary_text}",
            id=str(uuid.uuid4())
        )
        
        # 3. Post-Compact Context Restoration (重建核心状态)
        restored_context_msg = _build_post_compact_restoration(to_compact)
        
        new_messages = system_msgs + [summary_msg]
        if restored_context_msg:
            new_messages.append(restored_context_msg)
            
        new_messages.extend(recent_kept)
        
        _consecutive_failures = 0
        logger.info("Auto-Compact 成功，上下文已大幅压缩并恢复核心状态。")
        
        # 由于这是硬性写入，返回之后 `coordinator` 会把这个 new_messages 重新写入 State（如果可以的话），
        # 否则只在本次 API 请求中生效。
        return AutoCompactResult(was_compacted=True, messages=new_messages)
        
    except Exception as e:
        _consecutive_failures += 1
        logger.exception("Auto-Compact 失败")
        return AutoCompactResult(was_compacted=False, messages=messages, error=str(e))


async def _generate_global_summary(messages: List[BaseMessage]) -> str:
    from server.agent.llm_factory import get_planner_llm
    llm = get_planner_llm()
    
    # 抽取部分核心信息供总结，如果太大可能连总结模型也超长
    # 因为我们的 to_compact 可能是已经被 collapse 过的，包含 <collapsed> 标签
    prompt = "请作为系统核心记忆压缩模块，将以下用户与 AI 的完整交互历史（包含已经过局部折叠的片段）进行全局、系统性总结。必须保留所有用户未满足的需求、关键报错信息、和重要文件路径。\n\n"
    
    for m in messages:
        text = str(m.content)
        if len(text) > 2000:
            text = text[:1000] + "\n...(truncated)...\n" + text[-1000:]
        prompt += f"[{m.type}]: {text}\n"
        
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    return resp.content


def _build_post_compact_restoration(compacted_msgs: List[BaseMessage]) -> BaseMessage:
    """
    后置压缩恢复：找出最近查看或编辑过的文件，把它们的路径列出来，
    如果可能，这里甚至可以直接调用读取文件内容，但为了轻量化，我们只恢复“最近交互的文件列表”。
    """
    recent_files = set()
    for m in compacted_msgs:
        # 简单 heuristic: 搜索工具调用的 args
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                if name in ["read_local_file", "edit_local_file"]:
                    if "file_path" in args:
                        recent_files.add(args["file_path"])
                        
    if not recent_files:
        return None
        
    content = "【自动恢复的上下文】\n由于历史记录被压缩，系统为您恢复了最近操作过的关键文件路径，以便您随时重新读取：\n"
    for f in list(recent_files)[-5:]:
        content += f"- {f}\n"
        
    return SystemMessage(content=content, id=str(uuid.uuid4()))
