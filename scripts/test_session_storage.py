"""
测试脚本：验证峨眉山文旅 - JSONL 会话持久化机制 (session_storage.py)
"""
import os
import sys
import asyncio

# 将项目根目录加进 python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage
from server.agent.session_storage import (
    global_session_storage, 
    record_transcript, 
    load_transcript_file,
    get_session_transcript_path
)

async def main():
    print("=" * 60)
    print(" 验证基于 Claude Code 架构 of JSONL 纯增量会话日志写入")
    print("=" * 60)
    
    session_id = "test_persistence_001"
    file_path = get_session_transcript_path(session_id)
    
    if os.path.exists(file_path):
        os.remove(file_path)

    # 1. 模拟第一轮对话
    print("\n1. 模拟第一轮对话...")
    msg1 = HumanMessage(content="你好，我想去峨眉山", id="uuid-1")
    msg2 = AIMessage(content="好的，请问有什么特殊偏好吗？", id="uuid-2")
    
    await record_transcript(session_id, [msg1, msg2])
    # 强制刷盘
    await global_session_storage.flush()
    
    # 验证文件是否生成
    assert os.path.exists(file_path), "JSONL 日志文件未生成！"
    with open(file_path, "r", encoding="utf-8") as f:
        lines1 = f.readlines()
        print(f"  当前日志行数: {len(lines1)}")
        assert len(lines1) == 2, f"期望 2 行，实际 {len(lines1)} 行"
        assert '"parentUuid"' not in lines1[0], "第一条记录不应包含 parentUuid"
        assert '"parentUuid":"uuid-1"' in lines1[1] or '"parentUuid": "uuid-1"' in lines1[1]

    # 2. 模拟第二轮对话（全量 messages 传入，验证增量排重）
    print("\n2. 模拟第二轮对话（验证增量排重）...")
    msg3 = HumanMessage(content="我带了小孩", id="uuid-3")
    msg4 = AIMessage(content="带小孩的话推荐坐观光车去清音阁。", id="uuid-4")
    
    # 传入全量 [msg1, msg2, msg3, msg4]
    await record_transcript(session_id, [msg1, msg2, msg3, msg4])
    await global_session_storage.flush()
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines2 = f.readlines()
        print(f"  当前日志行数: {len(lines2)}")
        assert len(lines2) == 4, f"期望 4 行（增量追加2行），实际 {len(lines2)} 行"
        assert '"parentUuid":"uuid-2"' in lines2[2] or '"parentUuid": "uuid-2"' in lines2[2]
        assert '"parentUuid":"uuid-3"' in lines2[3] or '"parentUuid": "uuid-3"' in lines2[3]

    # 3. 验证通过 parentUuid 的反向链表重构功能
    print("\n3. 模拟加载历史会话记录 (load_transcript_file)...")
    rebuilt_msgs = await load_transcript_file(session_id)
    
    print(f"  重建的消息数量: {len(rebuilt_msgs)}")
    assert len(rebuilt_msgs) == 4
    assert rebuilt_msgs[0].content == "你好，我想去峨眉山"
    assert rebuilt_msgs[-1].content == "带小孩的话推荐坐观光车去清音阁。"
    print("  重建的内容完全匹配。")

    # 清理
    if os.path.exists(file_path):
        os.remove(file_path)
    print("\n[SUCCESS] Session Storage (JSONL Transcript) 验证全部通过！")

if __name__ == "__main__":
    asyncio.run(main())
