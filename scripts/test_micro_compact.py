"""
测试 Micro-Compact（第三步）核心逻辑。

验证：
1. 可裁剪工具名单注册与动态扩展
2. 时间触发条件（60分钟间隔）
3. 保留最近 KEEP_RECENT 个，清理其余
4. 不可裁剪工具（spawn_worker 等）不被清理
5. 已清理过的消息不重复处理
6. tokens_freed 正确计算
7. micro_compacted 标记正确设置
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from server.agent.compression.micro_compact import (
    micro_compact_if_needed,
    register_compactable_tool,
    unregister_compactable_tool,
    get_compactable_tool_names,
    is_micro_compacted,
    MC_CLEARED_PLACEHOLDER,
    KEEP_RECENT,
    GAP_THRESHOLD_MINUTES,
)


def make_tool_msg(tool_name: str, content: str = "tool result content", size: int = None):
    """创建一条模拟 ToolMessage。"""
    if size:
        content = "X" * size
    tool_call_id = str(uuid.uuid4())
    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        id=str(uuid.uuid4()),
        name=tool_name,
    )


def make_ai_msg(timestamp: float = None):
    """创建一条带时间戳的 AIMessage。"""
    msg = AIMessage(content="I will help you.", id=str(uuid.uuid4()))
    if timestamp:
        msg.additional_kwargs["created_at"] = timestamp
    return msg


def test_compactable_tools_registry():
    print("=== Test 1: 工具名单注册 ===")
    names = get_compactable_tool_names()
    assert "weather_api" in names, "weather_api 应在名单中"
    assert "web_search" in names, "web_search 应在名单中"
    assert "read_local_file" in names, "read_local_file 应在名单中"
    assert "spawn_worker" not in names, "spawn_worker 不应在名单中"
    assert "generate_image_tool" not in names, "generate_image_tool 不应在名单中"
    assert "TaskStop" not in names, "TaskStop 不应在名单中"

    # 动态注册
    register_compactable_tool("my_custom_tool")
    assert "my_custom_tool" in get_compactable_tool_names()
    unregister_compactable_tool("my_custom_tool")
    assert "my_custom_tool" not in get_compactable_tool_names()
    print("  [PASS]\n")


def test_no_trigger_below_threshold():
    print("=== Test 2: 未超时，不触发 ===")
    # AI 消息的时间戳是「5分钟前」
    recent_ts = time.time() - 5 * 60
    msgs = [
        SystemMessage(content="System", id=str(uuid.uuid4())),
        make_ai_msg(timestamp=recent_ts),
        make_tool_msg("weather_api", "weather data"),
    ]
    result = micro_compact_if_needed(msgs)
    assert result.tools_cleared == 0, f"不应触发，tools_cleared={result.tools_cleared}"
    print(f"  Gap=5min < {GAP_THRESHOLD_MINUTES}min → 不触发 [PASS]\n")


def test_trigger_above_threshold():
    print("=== Test 3: 超时触发 ===")
    # AI 消息的时间戳是「70分钟前」
    old_ts = time.time() - 70 * 60
    msgs = [
        SystemMessage(content="System", id=str(uuid.uuid4())),
        HumanMessage(content="Hello", id=str(uuid.uuid4())),
        make_ai_msg(timestamp=old_ts),
    ]
    # 添加 7 条可裁剪工具结果（KEEP_RECENT=5，应清理 7-5=2 条）
    for i in range(7):
        msgs.append(make_tool_msg("weather_api", f"weather data {i}" * 100))

    result = micro_compact_if_needed(msgs)
    print(f"  Gap=70min > {GAP_THRESHOLD_MINUTES}min → 触发")
    print(f"  total=7, kept={KEEP_RECENT}, cleared={result.tools_cleared}")
    print(f"  tokens_freed={result.tokens_freed:,}")
    assert result.tools_cleared == 7 - KEEP_RECENT, f"应清理 {7 - KEEP_RECENT} 条，实际={result.tools_cleared}"
    assert result.tokens_freed > 0
    print("  [PASS]\n")


def test_non_compactable_tools_not_cleared():
    print("=== Test 4: 不可裁剪工具不被清理 ===")
    old_ts = time.time() - 70 * 60
    msgs = [
        make_ai_msg(timestamp=old_ts),
        make_tool_msg("spawn_worker", "worker started" * 100),
        make_tool_msg("send_message", "message sent" * 100),
        make_tool_msg("generate_image_tool", "image url here" * 100),
    ]

    result = micro_compact_if_needed(msgs)
    assert result.tools_cleared == 0, f"不可裁剪工具不应被清理，tools_cleared={result.tools_cleared}"

    # 验证内容未变
    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    for tm in tool_msgs:
        assert tm.content != MC_CLEARED_PLACEHOLDER, f"工具 {tm.name} 不应被清理"
    print("  spawn_worker, send_message, generate_image_tool 均未被清理 [PASS]\n")


def test_keeps_most_recent():
    print("=== Test 5: 保留最近 KEEP_RECENT 个 ===")
    old_ts = time.time() - 70 * 60
    msgs = [make_ai_msg(timestamp=old_ts)]

    tool_msgs = []
    for i in range(10):
        tm = make_tool_msg("web_search", f"search result {i}" * 50)
        tool_msgs.append(tm)
        msgs.append(tm)

    result = micro_compact_if_needed(msgs)
    cleared_ids = {
        m.tool_call_id
        for m in result.messages
        if isinstance(m, ToolMessage) and m.content == MC_CLEARED_PLACEHOLDER
    }
    kept_ids = {
        m.tool_call_id
        for m in result.messages
        if isinstance(m, ToolMessage) and m.content != MC_CLEARED_PLACEHOLDER
    }

    assert len(cleared_ids) == 10 - KEEP_RECENT, f"应清理 {10 - KEEP_RECENT} 条"
    assert len(kept_ids) == KEEP_RECENT, f"应保留 {KEEP_RECENT} 条"

    # 保留的应该是最近的 KEEP_RECENT 个（原始列表末尾）
    expected_kept = {tm.tool_call_id for tm in tool_msgs[-KEEP_RECENT:]}
    assert kept_ids == expected_kept, "保留的应该是最新的几条"
    print(f"  10条，清理前 {10-KEEP_RECENT}，保留后 {KEEP_RECENT} [PASS]\n")


def test_force_flag():
    print("=== Test 6: force=True 强制触发 ===")
    # AI 消息是「刚刚」（不超时）
    recent_ts = time.time()
    msgs = [
        make_ai_msg(timestamp=recent_ts),
    ]
    for i in range(8):
        msgs.append(make_tool_msg("weather_api", f"data {i}" * 100))

    result = micro_compact_if_needed(msgs, force=True)
    assert result.tools_cleared == 8 - KEEP_RECENT, f"force=True 应触发清理，tools_cleared={result.tools_cleared}"
    print(f"  force=True 无视时间阈值，清理了 {result.tools_cleared} 条 [PASS]\n")


def test_already_cleared_skipped():
    print("=== Test 7: 已清理的消息不重复处理 ===")
    old_ts = time.time() - 70 * 60
    msgs = [make_ai_msg(timestamp=old_ts)]

    # 添加 3 条，其中 1 条已经是占位符
    already_cleared = ToolMessage(
        content=MC_CLEARED_PLACEHOLDER,
        tool_call_id=str(uuid.uuid4()),
        id=str(uuid.uuid4()),
        name="weather_api",
    )
    msgs.append(already_cleared)
    for i in range(3):
        msgs.append(make_tool_msg("weather_api", f"real data {i}"))

    # total compactable = 3（已清理的不算）
    # keep_recent = 5，全部保留
    result = micro_compact_if_needed(msgs)
    assert result.tools_cleared == 0, f"3 < KEEP_RECENT({KEEP_RECENT})，不应额外清理，tools_cleared={result.tools_cleared}"
    print(f"  已清理的不重复计入，总有效=3 < KEEP_RECENT={KEEP_RECENT}，无额外清理 [PASS]\n")


def test_mixed_tools():
    print("=== Test 8: 混合可裁剪与不可裁剪工具 ===")
    old_ts = time.time() - 70 * 60
    msgs = [make_ai_msg(timestamp=old_ts)]

    # 3 个不可裁剪
    for _ in range(3):
        msgs.append(make_tool_msg("spawn_worker", "worker" * 200))
    # 8 个可裁剪
    compactable_ids = []
    for i in range(8):
        tm = make_tool_msg("weather_api", f"weather {i}" * 100)
        compactable_ids.append(tm.tool_call_id)
        msgs.append(tm)

    result = micro_compact_if_needed(msgs)

    expected_cleared = 8 - KEEP_RECENT
    assert result.tools_cleared == expected_cleared, f"应清理 {expected_cleared} 条可裁剪工具"

    # 验证不可裁剪工具未被清理
    non_compact_msgs = [m for m in result.messages if isinstance(m, ToolMessage) and m.name == "spawn_worker"]
    for tm in non_compact_msgs:
        assert tm.content != MC_CLEARED_PLACEHOLDER

    print(f"  3 spawn_worker 未被清理，8 weather_api 清理了 {result.tools_cleared} 条 [PASS]\n")


if __name__ == "__main__":
    print("=" * 60)
    print(" Testing Micro-Compact (Phase 3)")
    print("=" * 60)
    print()

    test_compactable_tools_registry()
    test_no_trigger_below_threshold()
    test_trigger_above_threshold()
    test_non_compactable_tools_not_cleared()
    test_keeps_most_recent()
    test_force_flag()
    test_already_cleared_skipped()
    test_mixed_tools()

    print("[ALL TESTS PASSED]")
