"""
测试脚本：验证峨眉山文旅记忆系统后台自动提取与写入组件（extractor.py）。
"""
import os
import sys
import asyncio

# 将项目根目录加进 python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.memory.manager import MemoryManager, INDEX_FILENAME
from server.memory.extractor import extract_travel_memories
from langchain_core.messages import HumanMessage, AIMessage

# --- Mock LangGraph State & App ---
class MockState:
    def __init__(self, messages):
        self.values = {"messages": messages}

class MockApp:
    def __init__(self, messages):
        self.messages = messages
        
    async def aget_state(self, config):
        return MockState(self.messages)


async def main():
    print("=" * 60)
    print(" 开始测试峨眉山文旅后台记忆提取系统")
    print("=" * 60)

    # 1. 初始化测试路径
    test_memory_dir = os.path.join(".data", "memory")
    manager = MemoryManager(memory_dir=test_memory_dir)
    print(f"1. 成功初始化测试环境，记忆目录为: {test_memory_dir}")

    # 清空历史测试文件
    if os.path.exists(test_memory_dir):
        for f in os.listdir(test_memory_dir):
            if f != INDEX_FILENAME:
                os.remove(os.path.join(test_memory_dir, f))
    manager.ensure_memory_dir()

    # 2. 模拟一轮典型的游客-智能体对话（包含硬性画像特征和偏好）
    messages = [
        HumanMessage(content="你好，智能体。我今天第一次来峨眉山，我带了我75岁的奶奶一起来。"),
        AIMessage(content="您好！欢迎您和奶奶来到秀丽的峨眉山！请问奶奶的身体状态如何？有什么游玩偏好吗？我好为你们定制路线。"),
        HumanMessage(content="我奶奶平时不怎么运动，体力比较差，而且膝盖有点关节炎，走不了太多陡峭的上山石阶。对了，我们对看庙不太感兴趣，更想去看自然风景和可爱的野生猴子。"),
        AIMessage(content="收到！老人的膝盖痛和体力是个硬性限制，我会为您直接规避徒步攀爬的“九十九道拐”等高强度陡峭路线。我会极力推荐您通过景区观光车直达五显岗游览清音阁及自然生态猴区，这是一段平缓的游道，非常适合奶奶，不需要走陡峭石阶。我这就帮您规划详细行程。"),
    ]
    
    mock_app = MockApp(messages)
    print("\n2. 成功 Mock 会话历史：")
    print("  游客: '我奶奶75岁...体力比较差...膝盖有点关节炎...对看庙没兴趣...想看猴子和风景'")
    print("  智能体: '收到！规避陡峭攀爬路线...推荐平缓的清音阁猴区游线...'")

    # 3. 运行异步后台提取任务
    print("\n3. 开始触发后台提取（调用大模型 Structured Output 提取并持久化）...")
    await extract_travel_memories(mock_app, session_id="test_session_extractor", last_processed_len=0)

    # 4. 验证磁盘上生成的文件
    print("\n4. 开始检查磁盘文件...")
    files = [f for f in os.listdir(test_memory_dir) if f != INDEX_FILENAME]
    print(f"  检测到生成的记忆文件: {files}")

    assert len(files) > 0, "提取失败：未生成任何记忆文件！"

    # 打印生成的记忆内容
    for file in files:
        file_path = os.path.join(test_memory_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print("-" * 60)
        print(f" 文件名: {file}")
        print(content)
        print("-" * 60)
        
        # 简单断言检验
        assert "---" in content
        assert "type:" in content
        assert "description:" in content

    # 打印生成的索引
    print("\n5. 检查生成的记忆索引 MEMORY.md ...")
    index_content = manager.load_memory_index()
    print("-" * 60)
    print(index_content)
    print("-" * 60)

    # 清理测试生成的文件，保持工作区干净
    print("\n6. 清理测试产生的记忆文件...")
    for f in os.listdir(test_memory_dir):
        if f != INDEX_FILENAME:
            os.remove(os.path.join(test_memory_dir, f))
    manager._regenerate_index()
    print(" 峨眉山文旅记忆提取器 (Extractor) 验证成功，全部通过！\n")

if __name__ == "__main__":
    asyncio.run(main())
