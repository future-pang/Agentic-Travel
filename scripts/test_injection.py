"""
测试脚本：验证峨眉山文旅记忆检索注入系统 (coordinator_node 内的 BM25 检索注入)。
"""
import os
import sys
import asyncio
import json
from unittest.mock import patch, MagicMock

# 将项目根目录加进 python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.memory.manager import MemoryManager, INDEX_FILENAME
from server.agent.node.coordinator import coordinator_node
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

async def main():
    print("=" * 60)
    print(" 开始测试峨眉山文旅记忆检索注入系统")
    print("=" * 60)

    # 1. 准备测试路径及记忆数据
    test_memory_dir = os.path.join(".data", "memory")
    manager = MemoryManager(memory_dir=test_memory_dir)
    print(f"1. 初始化测试记忆数据于: {test_memory_dir}")

    # 清除旧记忆
    if os.path.exists(test_memory_dir):
        for f in os.listdir(test_memory_dir):
            if f != INDEX_FILENAME:
                os.remove(os.path.join(test_memory_dir, f))
    manager.ensure_memory_dir()

    # 写入旅行画像 (Persona)
    manager.save_memory_topic(
        filename="traveler_persona.md",
        content="奶奶今年75岁，膝盖患有轻微关节炎，体力极差，不能攀爬太高陡峭山坡，但平缓山路可以步行。",
        memory_type="persona",
        description="同行有75岁膝盖关节炎的老人，体能差"
    )
    # 写入游玩偏好 (Preference)
    manager.save_memory_topic(
        filename="traveler_preference.md",
        content="游客极度偏好秀丽的自然山水、溪流以及峨眉山野生猴区；对宏大的寺庙与佛教文化兴趣一般；交通工具优先选择景区大巴与索道缆车。",
        memory_type="preference",
        description="喜欢风光和野生猴区，对寺庙文化一般，优先索道观光车"
    )
    print("  已成功预置：traveler_persona.md、traveler_preference.md 两项测试记忆。")

    # 2. 构造游客问题 (触发 BM25 匹配)
    user_query = "我们带老人家怎么在峨眉山游览合适？奶奶想看风景和猴子，有什么好玩的路线？"
    messages_history = [HumanMessage(content=user_query)]
    
    # 模拟 LangGraph AgentState
    state = {
        "messages": messages_history,
        "session_id": "test_session_injection",
        "user_query": user_query,
        "enable_web_search": False
    }
    print(f"\n2. 模拟游客提问: '{user_query}'")

    # 3. Mock 协调员的 LLM 调用，拦截拼装后的消息参数进行断言
    captured_messages = []
    
    async def mock_ainvoke(messages_list, **kwargs):
        nonlocal captured_messages
        captured_messages = messages_list
        return AIMessage(content="[Mocked Response]")

    # 用 Mock 拦截 _get_llm_with_tools
    mock_llm = MagicMock()
    mock_llm.ainvoke = mock_ainvoke
    
    print("\n3. 开始拦截 coordinator_node LLM 消息拼装...")
    with patch("server.agent.node.coordinator._get_llm_with_tools", return_value=mock_llm):
        await coordinator_node(state)

    # 4. 验证注入的消息结构和 JSON 格式内容
    print("\n4. 验证注入的消息体结构与 JSON 内容:")
    print(f"  拼装后的总消息项数: {len(captured_messages)}")
    
    # captured_messages 应该是：[system_msg, memory_system_msg, memory_context_msg, user_msg]
    assert len(captured_messages) == 4, "消息拼装结构错误！"
    
    memory_msg = captured_messages[2]
    assert isinstance(memory_msg, SystemMessage), "注入的记忆消息必须是 SystemMessage！"
    
    msg_content = memory_msg.content
    print("-" * 60)
    print(" 注入的 Traveler Memory Context 消息体:")
    print(msg_content)
    print("-" * 60)

    # 4.1 校验不含任何 XML 标签
    assert "<" not in msg_content and ">" not in msg_content, "违背了 '整个项目不要用 XML 格式' 的规范！"
    
    # 4.2 校验包含 Markdown JSON code block 格式
    assert "Traveler Memory Context:" in msg_content
    assert "```json" in msg_content
    
    # 4.3 解析并校验 JSON 结构与检索出来的核心关键字
    json_text = msg_content.split("```json")[1].split("```")[0].strip()
    memory_json = json.loads(json_text)
    
    assert "recalled_memories" in memory_json
    assert "instruction" in memory_json
    
    recalled = memory_json["recalled_memories"]
    print(f"  成功召回了 {len(recalled)} 条匹配度最高的文旅记忆！")
    assert len(recalled) == 2, "应该精确匹配召回这两条记忆！"
    
    filenames = [m["filename"] for m in recalled]
    assert "traveler_persona.md" in filenames
    assert "traveler_preference.md" in filenames
    print("  [校验成功] 正确在首位匹配到了 'traveler_persona.md' 和 'traveler_preference.md'！")

    # 5. 清理测试文件，保持工作区清爽
    print("\n5. 清理测试生成的文件...")
    for f in os.listdir(test_memory_dir):
        if f != INDEX_FILENAME:
            os.remove(os.path.join(test_memory_dir, f))
    manager._regenerate_index()
    
    print(" 峨眉山文旅记忆检索注入系统 (Injection) 验证成功，全部通过！\n")

if __name__ == "__main__":
    asyncio.run(main())
