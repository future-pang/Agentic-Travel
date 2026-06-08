"""
测试脚本：验证峨眉山文旅智能体记忆系统核心组件（MemoryManager）。
"""
import os
import sys
import time

# 将项目根目录加进 python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.memory import MemoryManager

def main():
    print("=" * 60)
    print(" 开始测试峨眉山文旅记忆系统")
    print("=" * 60)

    # 1. 初始化管理器，定制专用测试路径以便验证
    test_memory_dir = os.path.join(".data", "test_memory")
    manager = MemoryManager(memory_dir=test_memory_dir)
    print(f"1. 成功初始化 MemoryManager，测试目录为: {test_memory_dir}")

    # 清理历史测试数据
    if os.path.exists(test_memory_dir):
        for f in os.listdir(test_memory_dir):
            os.remove(os.path.join(test_memory_dir, f))
    manager.ensure_memory_dir()

    # 2. 模拟写入不同类型的旅程记忆
    print("\n2. 开始写入旅程记忆主题...")
    
    # 2.1 基础画像 (persona)
    persona_content = (
        "随行人员：带有1名60岁的老人（有轻微膝盖磨损，无法长时间爬山）。\n"
        "体力状态：用户本人体力中等偏上，但老人体力较弱。\n"
        "身体情况：老人恐高，需避开极其险峻的栈道。"
    )
    msg1 = manager.save_memory_topic(
        filename="traveler_persona.md",
        content=persona_content,
        memory_type="persona",
        description="随行有恐高、膝盖不太好的老人"
    )
    print(f"  [写入成功] traveler_persona.md -> {msg1}")

    # 2.2 游玩偏好 (preference)
    pref_content = (
        "景点偏好：对佛教寺庙文化兴趣一般，极度偏好秀丽的自然风光、瀑布和峨眉山野生猴区。\n"
        "出行方式：优先选择景区观光车与索道，尽量减少陡峭山路的步行攀爬。"
    )
    msg2 = manager.save_memory_topic(
        filename="travel_preference.md",
        content=pref_content,
        memory_type="preference",
        description="偏好自然风光与猴区，抗拒长时间徒步，对寺庙兴趣一般"
    )
    print(f"  [写入成功] travel_preference.md -> {msg2}")

    # 2.3 即时上下文 (realtime_ctx)
    ctx_content = (
        "当前位置：雷洞坪车站（刚从报国寺乘坐观光车抵达）。\n"
        "实时天气：雷洞坪目前大雾，气温 8 度，有毛毛细雨。\n"
        "行李状态：重型双肩包已寄存在报国寺，目前仅携带轻便背包和雨衣。\n"
        "已打卡点：报国寺。\n"
        "剩余游玩时间：今天还剩 4 小时，计划在金顶住宿一晚。"
    )
    msg3 = manager.save_memory_topic(
        filename="trip_status.md",
        content=ctx_content,
        memory_type="realtime_ctx",
        description="当前位于雷洞坪且天气大雾，计划今晚宿金顶"
    )
    print(f"  [写入成功] trip_status.md -> {msg3}")

    # 2.4 实时反馈 (feedback)
    feedback_content = (
        "用户反馈：刚刚在雷洞坪步行了1公里，老人觉得山路太滑，而且气温很低，膝盖开始隐隐作痛。\n"
        "身体反馈：老人体力消耗严重，已经开始有些气喘。"
    )
    msg4 = manager.save_memory_topic(
        filename="stamina_feedback.md",
        content=feedback_content,
        memory_type="feedback",
        description="老人因天气冷且山路滑，体力消耗大、膝盖疼痛"
    )
    print(f"  [写入成功] stamina_feedback.md -> {msg4}")

    # 3. 验证 MEMORY.md 索引的自动生成与内容
    print("\n3. 验证 MEMORY.md 索引生成...")
    index_content = manager.load_memory_index()
    print("-" * 60)
    print(index_content)
    print("-" * 60)

    # 断言校验索引中包含对应的关键字
    assert "旅程记忆列表" in index_content
    assert "[基础画像]" in index_content
    assert "[游玩偏好]" in index_content
    assert "[即时上下文]" in index_content
    assert "[实时反馈]" in index_content
    assert "traveler_persona.md" in index_content
    assert "stamina_feedback.md" in index_content

    # 4. 验证读取指定主题
    print("\n4. 验证读取特定主题内容 (trip_status.md)...")
    topic_detail = manager.read_memory_topic("trip_status.md")
    print("-" * 60)
    print(topic_detail)
    print("-" * 60)
    assert "---" in topic_detail
    assert "type: realtime_ctx" in topic_detail
    assert "当前位置：雷洞坪车站" in topic_detail

    # 4.5 验证轻量头部扫描 (scan_memory_headers) 与陈旧度检测 (check_stale_memories)
    print("\n4.5 验证轻量头部扫描与陈旧度检测...")
    headers = manager.scan_memory_headers()
    print(f"  已扫描到 {len(headers)} 条记忆头部。")
    assert len(headers) == 4
    filenames = [h["filename"] for h in headers]
    assert "trip_status.md" in filenames
    assert "stamina_feedback.md" in filenames
    assert "travel_preference.md" in filenames
    assert "traveler_persona.md" in filenames

    # 检查其中一个属性
    h_trip = next(h for h in headers if h["filename"] == "trip_status.md")
    assert h_trip["type"] == "realtime_ctx"
    assert "雷洞坪" in h_trip["preview"]

    # 4.6 验证陈旧度检测
    print("\n4.6 验证陈旧度检测...")
    persona_path = os.path.join(test_memory_dir, "traveler_persona.md")
    eight_days_ago = time.time() - 8 * 86400
    os.utime(persona_path, (eight_days_ago, eight_days_ago))
    
    stale_memories = manager.check_stale_memories(days=7)
    print(f"  检测到陈旧记忆数: {len(stale_memories)}")
    assert len(stale_memories) == 1
    assert stale_memories[0]["filename"] == "traveler_persona.md"
    assert stale_memories[0]["age_days"] == 8

    # 5. 清理测试文件
    print("\n5. 清理测试文件...")
    for f in os.listdir(test_memory_dir):
        os.remove(os.path.join(test_memory_dir, f))
    os.rmdir(test_memory_dir)
    print(" 记忆系统核心功能全部通过校验！测试成功！\n")

if __name__ == "__main__":
    main()
