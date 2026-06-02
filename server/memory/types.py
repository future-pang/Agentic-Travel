"""
峨眉山文旅智能体 记忆系统 - 类型与结构体定义模块。
"""

TRAVEL_MEMORY_TYPES = [
    'persona',      # 基础画像：同行人员、特殊人群、预算区间、物理状态（如体力、是否恐高）
    'preference',   # 游玩偏好：偏好人文/自然、餐饮禁忌、偏好的出行方式（索道 vs 徒步）
    'realtime_ctx', # 即时上下文：当前位置、天气、行李状态、已打卡点、剩余游玩时间
    'feedback'      # 实时反馈：对上一个景点的评价（太累了、不好看、排队太长），用于动态调整后续行程
]

def is_valid_memory_type(mtype: str) -> bool:
    """验证是否为合法的旅程记忆类型。"""
    return mtype in TRAVEL_MEMORY_TYPES

def get_type_display_name(mtype: str) -> str:
    """获取记忆类型的中文显示名称。"""
    display_names = {
        'persona': '基础画像',
        'preference': '游玩偏好',
        'realtime_ctx': '即时上下文',
        'feedback': '实时反馈'
    }
    return display_names.get(mtype, mtype)
