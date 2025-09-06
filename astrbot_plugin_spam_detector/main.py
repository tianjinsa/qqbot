import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig
import astrbot.api.message_components as Comp
from astrbot.api.provider import ProviderRequest


@register("astrbot_plugin_spam_detector", "AstrBot Dev Team", "智能防推销插件，使用AI检测并处理推销信息", "1.0.0", "https://github.com/AstrBotDevs/astrbot_plugin_spam_detector")
class SpamDetectorPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 用户消息历史缓存 (用户ID -> 消息列表)
        self.user_message_history: Dict[str, List[Dict[str, Any]]] = {}
        
    async def initialize(self):
        """插件初始化"""
        logger.info("防推销插件已启动")
        
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值，带默认值支持"""
        return self.config.get(key, default)
    
    def _is_user_whitelisted(self, user_id: str) -> bool:
        """检查用户是否在白名单中"""
        whitelist = self._get_config_value("WHITELIST_USERS", [])
        if isinstance(whitelist, str):
            # 如果是字符串，按逗号分割
            whitelist = [uid.strip() for uid in whitelist.split(",") if uid.strip()]
        return user_id in whitelist
    
    def _store_user_message(self, user_id: str, message_content: str, timestamp: float, message_id: str = ""):
        """存储用户消息到历史记录"""
        if user_id not in self.user_message_history:
            self.user_message_history[user_id] = []
        
        self.user_message_history[user_id].append({
            "content": message_content,
            "timestamp": timestamp,
            "message_id": message_id
        })
        
        # 清理过期消息（超过1小时的消息）
        cutoff_time = timestamp - 3600  # 1小时前
        self.user_message_history[user_id] = [
            msg for msg in self.user_message_history[user_id] 
            if msg["timestamp"] > cutoff_time
        ]
    
    def _get_user_recent_messages(self, user_id: str, last_minutes: int) -> List[str]:
        """获取用户在指定时间内的所有消息"""
        if user_id not in self.user_message_history:
            return []
        
        cutoff_time = time.time() - (last_minutes * 60)
        recent_messages = [
            msg["content"] for msg in self.user_message_history[user_id]
            if msg["timestamp"] > cutoff_time
        ]
        return recent_messages
    
    async def _get_context_messages(self, event: AstrMessageEvent, count: int) -> List[str]:
        """获取上下文消息（这里简化实现，实际可能需要调用平台API获取历史消息）"""
        # 由于API限制，这里暂时返回空列表
        # 实际实现中可能需要调用特定平台的API来获取历史消息
        return []
    
    async def _extract_image_content(self, image_urls: List[str]) -> str:
        """使用视觉模型提取图片内容"""
        if not image_urls:
            return ""
        
        try:
            # 使用AstrBot的LLM接口进行图片内容识别
            provider = self.context.get_using_provider()
            if not provider:
                logger.warning("未找到可用的LLM提供商，无法处理图片内容")
                return ""
            
            response = await provider.text_chat(
                prompt="请描述这张图片的主要内容，特别是如果有文字请提取出来。",
                image_urls=image_urls,
                system_prompt="你是一个图片内容识别助手，请客观描述图片内容。"
            )
            
            if response and response.completion_text:
                return response.completion_text
        except Exception as e:
            logger.error(f"图片内容提取失败: {e}")
        
        return ""
    
    async def _is_spam_message(self, message_content: str, context_messages: List[str], image_content: str = "") -> bool:
        """使用LLM判断是否为推销消息"""
        try:
            provider = self.context.get_using_provider()
            if not provider:
                logger.warning("未找到可用的LLM提供商，无法进行推销检测")
                return False
            
            # 合并消息内容
            full_content = message_content
            if image_content:
                full_content += f"\n\n图片内容：{image_content}"
            
            # 构建上下文
            context_text = ""
            if context_messages:
                context_text = f"\n\n最近的对话上下文：\n" + "\n".join(context_messages)
            
            # 获取系统提示词
            system_prompt = self._get_config_value("LLM_SYSTEM_PROMPT", 
                """你是一个专业的推销信息检测助手。请分析给定的消息内容，判断它是否是推销信息。

推销信息的特征包括但不限于：
1. 销售产品或服务
2. 包含价格、优惠、折扣等商业信息
3. 引导添加微信、QQ等联系方式进行交易
4. 推广某个商品、品牌或服务
5. 含有明显的营销意图

请只回答"是"或"否"，如果是推销信息回答"是"，如果不是推销信息回答"否"。""")
            
            prompt = f"请判断以下消息是否为推销信息：\n\n{full_content}{context_text}"
            
            response = await provider.text_chat(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            if response and response.completion_text:
                result = response.completion_text.strip().lower()
                # 判断LLM的回复
                return "是" in result or "yes" in result or "spam" in result
                
        except Exception as e:
            logger.error(f"推销检测失败: {e}")
        
        return False
    
    async def _handle_spam_message(self, event: AstrMessageEvent, user_id: str, user_name: str):
        """处理检测到的推销消息"""
        try:
            # 1. 获取用户最近的消息
            last_time = self._get_config_value("LAST_TIME", 5)
            recent_messages = self._get_user_recent_messages(user_id, last_time)
            
            # 2. 转发到管理员群
            admin_chat_id = self._get_config_value("ADMIN_CHAT_ID", "")
            if admin_chat_id:
                await self._forward_to_admin(admin_chat_id, user_name, user_id, recent_messages, event)
            
            # 3. 撤回原消息（如果平台支持）
            await self._try_recall_message(event)
            
            # 4. 发送警告消息
            alert_message = self._get_config_value("SPAM_ALERT_MESSAGE", 
                "⚠️ 检测到疑似推销信息，该消息已被处理。")
            yield event.plain_result(alert_message)
            
        except Exception as e:
            logger.error(f"处理推销消息时出错: {e}")
    
    async def _forward_to_admin(self, admin_chat_id: str, user_name: str, user_id: str, 
                              recent_messages: List[str], event: AstrMessageEvent):
        """转发消息到管理员群"""
        try:
            # 构建转发内容
            forward_content = f"🚨 推销检测报告\n"
            forward_content += f"用户: {user_name} ({user_id})\n"
            forward_content += f"平台: {event.get_platform_name()}\n"
            forward_content += f"群组: {event.get_group_id() or '私聊'}\n"
            forward_content += f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            forward_content += f"最近 {len(recent_messages)} 条消息:\n"
            
            for i, msg in enumerate(recent_messages, 1):
                forward_content += f"{i}. {msg}\n"
            
            # 构建统一消息来源标识符
            admin_unified_origin = f"{event.get_platform_name()}:group:{admin_chat_id}"
            
            # 发送到管理员群
            from astrbot.api.event import MessageChain
            message_chain = MessageChain().message(forward_content)
            await self.context.send_message(admin_unified_origin, message_chain)
            
        except Exception as e:
            logger.error(f"转发到管理员群失败: {e}")
    
    async def _try_recall_message(self, event: AstrMessageEvent):
        """尝试撤回消息（如果平台支持）"""
        try:
            platform_name = event.get_platform_name()
            if platform_name == "aiocqhttp":
                # 对于aiocqhttp平台，尝试撤回消息
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    payloads = {
                        "message_id": event.message_obj.message_id,
                    }
                    await client.api.call_action('delete_msg', **payloads)
                    logger.info(f"已撤回推销消息: {event.message_obj.message_id}")
        except Exception as e:
            logger.warning(f"撤回消息失败（可能平台不支持）: {e}")
    
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """监听群聊消息"""
        try:
            user_id = event.get_sender_id()
            user_name = event.get_sender_name()
            message_content = event.message_str
            timestamp = time.time()
            
            # 白名单检查
            if self._is_user_whitelisted(user_id):
                logger.debug(f"用户 {user_id} 在白名单中，跳过检测")
                return
            
            # 存储用户消息
            self._store_user_message(user_id, message_content, timestamp, 
                                   getattr(event.message_obj, 'message_id', ''))
            
            # 提取图片内容
            image_urls = []
            for msg_comp in event.get_messages():
                if isinstance(msg_comp, Comp.Image):
                    if hasattr(msg_comp, 'url') and msg_comp.url:
                        image_urls.append(msg_comp.url)
                    elif hasattr(msg_comp, 'file') and msg_comp.file:
                        image_urls.append(msg_comp.file)
            
            image_content = ""
            if image_urls:
                image_content = await self._extract_image_content(image_urls)
            
            # 获取上下文消息
            context_count = self._get_config_value("CONTEXT_MESSAGE_COUNT", 1)
            context_messages = await self._get_context_messages(event, context_count)
            
            # 检测是否为推销消息
            is_spam = await self._is_spam_message(message_content, context_messages, image_content)
            
            if is_spam:
                logger.info(f"检测到推销消息，用户: {user_name} ({user_id})")
                await self._handle_spam_message(event, user_id, user_name)
                
        except Exception as e:
            logger.error(f"处理群聊消息时出错: {e}")
    
    @filter.command("spam_whitelist", alias={"垃圾白名单", "推销白名单"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def manage_whitelist(self, event: AstrMessageEvent, action: str = "", user_id: str = ""):
        """管理推销检测白名单"""
        try:
            if action == "add" and user_id:
                # 添加到白名单
                whitelist = self._get_config_value("WHITELIST_USERS", [])
                if isinstance(whitelist, str):
                    whitelist = [uid.strip() for uid in whitelist.split(",") if uid.strip()]
                
                if user_id not in whitelist:
                    whitelist.append(user_id)
                    self.config["WHITELIST_USERS"] = whitelist
                    self.config.save_config()
                    yield event.plain_result(f"✅ 用户 {user_id} 已添加到推销检测白名单")
                else:
                    yield event.plain_result(f"ℹ️ 用户 {user_id} 已在白名单中")
                    
            elif action == "remove" and user_id:
                # 从白名单移除
                whitelist = self._get_config_value("WHITELIST_USERS", [])
                if isinstance(whitelist, str):
                    whitelist = [uid.strip() for uid in whitelist.split(",") if uid.strip()]
                
                if user_id in whitelist:
                    whitelist.remove(user_id)
                    self.config["WHITELIST_USERS"] = whitelist
                    self.config.save_config()
                    yield event.plain_result(f"✅ 用户 {user_id} 已从推销检测白名单移除")
                else:
                    yield event.plain_result(f"ℹ️ 用户 {user_id} 不在白名单中")
                    
            elif action == "list":
                # 查看白名单
                whitelist = self._get_config_value("WHITELIST_USERS", [])
                if isinstance(whitelist, str):
                    whitelist = [uid.strip() for uid in whitelist.split(",") if uid.strip()]
                
                if whitelist:
                    yield event.plain_result(f"📋 推销检测白名单:\n" + "\n".join(f"- {uid}" for uid in whitelist))
                else:
                    yield event.plain_result("📋 推销检测白名单为空")
            else:
                yield event.plain_result(
                    "📝 推销检测白名单管理命令:\n"
                    "/spam_whitelist add <用户ID> - 添加用户到白名单\n"
                    "/spam_whitelist remove <用户ID> - 从白名单移除用户\n"
                    "/spam_whitelist list - 查看白名单"
                )
                
        except Exception as e:
            logger.error(f"管理白名单时出错: {e}")
            yield event.plain_result("❌ 操作失败，请检查日志")
    
    @filter.command("spam_test", alias={"推销测试"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def test_spam_detection(self, event: AstrMessageEvent, message: str):
        """测试推销检测功能"""
        try:
            is_spam = await self._is_spam_message(message, [], "")
            result = "✅ 是推销信息" if is_spam else "❌ 不是推销信息"
            yield event.plain_result(f"🔍 推销检测结果: {result}\n测试消息: {message}")
        except Exception as e:
            logger.error(f"测试推销检测时出错: {e}")
            yield event.plain_result("❌ 测试失败，请检查日志")
    
    async def terminate(self):
        """插件卸载时的清理工作"""
        logger.info("防推销插件已停止")
        self.user_message_history.clear()
