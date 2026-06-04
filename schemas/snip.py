"""
Snip 压缩功能的输入/输出 Schema。

Snip 是上下文窗口压缩的第二层，负责将对话历史开头的老消息物理移除，
并插入边界标记通知模型「这之前的内容已被压缩」。
"""
from typing import Optional
from pydantic import BaseModel, Field


class SnipToolInputSpec(BaseModel):
    """SnipTool 的入参 Schema。"""
    from_id: Optional[str] = Field(
        default=None,
        description="要 Snip 范围的起始消息 short_id（6位 base36）。为空时从最早的非系统消息开始。"
    )
    to_id: str = Field(
        description="要 Snip 范围的结束消息 short_id（6位 base36，含），该消息之前的所有内容都会被移除。"
    )


class SnipToolOutputSpec(BaseModel):
    """SnipTool 的出参 Schema。"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作描述或失败原因")
    tokens_freed: int = Field(default=0, description="释放的估算 Token 数")
    messages_removed: int = Field(default=0, description="移除的消息数量")
    error_code: Optional[int] = Field(default=None, description="失败时的错误码")
