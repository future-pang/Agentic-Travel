"""
Micro-Compact 模块：裁剪老的工具输出。

这是上下文窗口压缩的第三层：时间衰减策略。
核心思想：越老的工具结果越不重要，且 Prompt Cache 大概率已过期——
既然缓存没了，不如把旧工具结果清空，节省后续调用的输入 token。

触发条件：
  距离上一条 AI 消息（assistant）超过 GAP_THRESHOLD_MINUTES（默认 60 分钟），
  说明会话已冷启动，API 端的 Prompt Cache 基本已失效。

核心逻辑：
  1. 找到所有「可裁剪」工具结果（COMPACTABLE_TOOLS 名单内）
  2. 按调用时序排序，保留最新的 KEEP_RECENT 个（默认 5）
  3. 其余替换为占位字符串 '[Old tool result content cleared]'

不可裁剪的工具（结果不可重复）：
  - spawn_worker / send_message：Worker 的推理过程不可重复
  - TaskStop / SnipTool：控制类工具，结果无实质内容
  - generate_image_tool：图片生成不可重复

可裁剪的工具（结果可重新获取）：
  - 天气类：weather_api, weather_forecast_api, travel_advice_api, astronomy_api
  - 地图类：get_walking_plan, get_distance, search_around, get_static_map
  - 搜索类：web_search, search_knowledge_base
  - 文件类：read_local_file
  - 时间类：get_current_time（结果已过时）

设计说明：
  - 与 Claude Code 不同，我们没有 Anthropic cache_edits API，所以只实现
    「时间触发清理」路径（Path A），不实现 cache_edits 路径（Path B）。
  - 工具名单（COMPACTABLE_TOOLS）设计为可在运行时动态扩展，
    调用 register_compactable_tool() 即可注册新工具名。
"""

import time
from typing import List, Optional, Set
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

from utils.tokens import rough_estimation_for_messages
from utils.logger import get_logger

logger = get_logger("shiliu.micro_compact")

# ── 常量 ──────────────────────────────────────────────────────────────────────

# 时间阈值：距上一条 AI 消息超过此分钟数，认为 Prompt Cache 已失效
GAP_THRESHOLD_MINUTES = 60

# 每类可裁剪工具保留最近 N 个结果（保留最新的，清理其余的）
KEEP_RECENT = 5

# 裁剪后替换的占位字符串（与 Claude Code 保持一致）
MC_CLEARED_PLACEHOLDER = "[Old tool result content cleared]"

# ── 可裁剪工具名单（可动态注册）────────────────────────────────────────────────

# 核心规则：「可重新获取」的工具结果才能被裁剪。
# 不可裁剪的：spawn_worker, send_message（Worker 推理不可重复）
#             TaskStop, SnipTool（控制工具）
#             generate_image_tool（图片生成不可重复）
COMPACTABLE_TOOLS: Set[str] = {
    # 天气类 —— 可重新查询
    "weather_api",
    "weather_forecast_api",
    "travel_advice_api",
    "astronomy_api",
    # 地图类 —— 可重新查询
    "get_walking_plan",
    "get_distance",
    "search_around",
    "get_static_map",
    # 搜索类 —— 可重新搜索
    "web_search",
    "search_knowledge_base",
    # 文件类 —— 可重新读取（且第一步 tool_persistence 已持久化原始内容）
    "read_local_file",
    # 时间类 —— 旧的时间戳已无意义
    "get_current_time",
}


def register_compactable_tool(tool_name: str) -> None:
    """
    动态注册一个可裁剪工具。
    当项目新增工具时，在 tool_manager.py 里调用此函数即可，
    无需改动 micro_compact.py。
    """
    COMPACTABLE_TOOLS.add(tool_name)
    logger.debug("已注册可裁剪工具", tool_name=tool_name)


def unregister_compactable_tool(tool_name: str) -> None:
    """将一个工具从可裁剪名单中移除（防止误清重要工具）。"""
    COMPACTABLE_TOOLS.discard(tool_name)


# ── 核心逻辑 ──────────────────────────────────────────────────────────────────

class MicroCompactResult:
    """micro_compact_if_needed 的返回值。"""
    def __init__(
        self,
        messages: List[BaseMessage],
        tokens_freed: int = 0,
        tools_cleared: int = 0,
    ):
        self.messages = messages
        self.tokens_freed = tokens_freed    # 估算释放的 token 数
        self.tools_cleared = tools_cleared  # 实际清理的工具结果条数


def micro_compact_if_needed(
    messages: List[BaseMessage],
    gap_threshold_minutes: float = GAP_THRESHOLD_MINUTES,
    keep_recent: int = KEEP_RECENT,
    force: bool = False,
) -> MicroCompactResult:
    """
    时间触发的工具结果裁剪。

    Args:
        messages: 当前完整消息列表。
        gap_threshold_minutes: 距上一条 AI 消息超过多少分钟才触发（默认 60 分钟）。
        keep_recent: 每个可裁剪工具保留最近 N 个结果（默认 5）。
        force: 强制裁剪，忽略时间阈值检查（用于测试或手动触发）。

    Returns:
        MicroCompactResult(messages, tokens_freed, tools_cleared)
    """
    if not messages:
        return MicroCompactResult(messages)

    if not force:
        gap_minutes = _get_gap_since_last_ai_message(messages)
        if gap_minutes is None or gap_minutes < gap_threshold_minutes:
            return MicroCompactResult(messages)

    # 收集所有可裁剪的 ToolMessage 的 tool_call_id（按顺序排列）
    compactable_ids = _collect_compactable_tool_ids(messages)

    if not compactable_ids:
        return MicroCompactResult(messages)

    # 保留最新的 keep_recent 个，其余进入清理集合
    keep_set: Set[str] = set(compactable_ids[-keep_recent:])
    clear_set: Set[str] = {tid for tid in compactable_ids if tid not in keep_set}

    if not clear_set:
        return MicroCompactResult(messages)

    # 执行替换
    new_messages, tokens_freed, tools_cleared = _apply_micro_compact(messages, clear_set)

    if tools_cleared > 0:
        logger.info(
            "Micro-Compact 工具结果裁剪完成",
            extra={
                "compactable_total": len(compactable_ids),
                "kept": len(keep_set),
                "cleared": tools_cleared,
                "tokens_freed": tokens_freed,
            }
        )

    return MicroCompactResult(new_messages, tokens_freed, tools_cleared)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _get_gap_since_last_ai_message(messages: List[BaseMessage]) -> Optional[float]:
    """
    计算距离最后一条 AI 消息的时间差（分钟）。

    从 usage_metadata 或 additional_kwargs 中提取时间戳（若有），
    否则使用消息的创建顺序位置作为降级判断——
    如果消息列表里没有任何时间戳信息，返回 None（不触发）。
    """
    now = time.time()

    # 从最后往前找最近的 AI 消息，尝试提取时间戳
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue

        # 优先从 additional_kwargs 中取 created_at（我们在 to_transcript 时有写入）
        ts = None
        meta = getattr(msg, "additional_kwargs", {}) or {}
        ts = meta.get("created_at") or meta.get("timestamp")

        # 也可以从 response_metadata 中取
        if ts is None:
            response_meta = getattr(msg, "response_metadata", {}) or {}
            ts = response_meta.get("created_at") or response_meta.get("timestamp")

        if ts is not None:
            try:
                gap_seconds = now - float(ts)
                return gap_seconds / 60.0
            except (ValueError, TypeError):
                pass

    # 若完全没有时间戳信息，返回 None（不触发时间条件）
    return None


def _collect_compactable_tool_ids(messages: List[BaseMessage]) -> List[str]:
    """
    按消息顺序收集所有「可裁剪」工具结果的 tool_call_id。

    遍历 ToolMessage，通过 name 字段判断是否在 COMPACTABLE_TOOLS 名单内。
    已经是占位字符串（之前已清理过）的消息跳过。
    """
    compactable_ids: List[str] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        # 已经被清理过的跳过，避免重复处理
        if isinstance(msg.content, str) and msg.content == MC_CLEARED_PLACEHOLDER:
            continue

        # ToolMessage.name 字段存放工具名（LangGraph ToolNode 会自动填充）
        tool_name = getattr(msg, "name", None) or ""

        if tool_name in COMPACTABLE_TOOLS:
            compactable_ids.append(msg.tool_call_id)

    return compactable_ids


def _apply_micro_compact(
    messages: List[BaseMessage],
    clear_set: Set[str],
) -> tuple:
    """
    遍历消息列表，将 clear_set 中的 ToolMessage 内容替换为占位字符串。

    返回 (new_messages, tokens_freed, tools_cleared)
    """
    new_messages = []
    tokens_freed = 0
    tools_cleared = 0

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            new_messages.append(msg)
            continue

        if msg.tool_call_id not in clear_set:
            new_messages.append(msg)
            continue

        # 计算被清理的 token 数（清理前估算）
        original_tokens = rough_estimation_for_messages([msg])
        placeholder_tokens = len(MC_CLEARED_PLACEHOLDER) // 4  # 约 8 tokens

        tokens_freed += max(0, original_tokens - placeholder_tokens)
        tools_cleared += 1

        # 创建内容已替换的新 ToolMessage（不改变 tool_call_id 和 id，保持消息链完整）
        cleared_msg = ToolMessage(
            content=MC_CLEARED_PLACEHOLDER,
            tool_call_id=msg.tool_call_id,
            id=msg.id,
            name=getattr(msg, "name", None),
            additional_kwargs={
                **getattr(msg, "additional_kwargs", {}),
                "micro_compacted": True,
            }
        )
        new_messages.append(cleared_msg)

    return new_messages, tokens_freed, tools_cleared


def is_micro_compacted(msg: BaseMessage) -> bool:
    """判断一条 ToolMessage 是否已被 Micro-Compact 清理过。"""
    if not isinstance(msg, ToolMessage):
        return False
    meta = getattr(msg, "additional_kwargs", {}) or {}
    return meta.get("micro_compacted", False) is True


def get_compactable_tool_names() -> Set[str]:
    """返回当前可裁剪工具名单的只读副本（用于调试/测试）。"""
    return set(COMPACTABLE_TOOLS)
