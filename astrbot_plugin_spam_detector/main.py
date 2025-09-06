import asyncio
import json
import time
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiohttp

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig
import astrbot.api.message_components as Comp


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
        
    async def _make_openai_request(self, base_url: str, api_key: str, model_id: str, 
                                 messages: List[Dict], timeout: int = 30) -> Optional[str]:
        """发送OpenAI格式的HTTP请求"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            url = f"{base_url.rstrip('/')}/chat/completions"
            
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"模型API请求失败: {response.status} - {error_text}")
                        
        except asyncio.TimeoutError:
            logger.error(f"模型API请求超时: {timeout}秒")
        except Exception as e:
            logger.error(f"模型API请求异常: {e}")
        
        return None
        
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
    
    def _is_group_whitelisted(self, group_id: str) -> bool:
        """检查群聊是否在白名单中"""
        if not group_id:
            return False
        
        whitelist = self._get_config_value("WHITELIST_GROUPS", [])
        if isinstance(whitelist, str):
            # 如果是字符串，按逗号分割
            whitelist = [gid.strip() for gid in whitelist.split(",") if gid.strip()]
        
        # 如果白名单为空，则检测所有群聊
        if not whitelist:
            return True
        
        return group_id in whitelist
    
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
        """使用自定义视觉模型提取图片内容"""
        if not image_urls:
            return ""
        
        try:
            # 获取视觉模型配置
            vision_model_id = self._get_config_value("VISION_MODEL_ID", "gpt-4-vision-preview")
            vision_base_url = self._get_config_value("VISION_MODEL_BASE_URL", "https://api.openai.com/v1")
            vision_api_key = self._get_config_value("VISION_MODEL_API_KEY", "")
            timeout = self._get_config_value("MODEL_TIMEOUT", 30)
            
            if not vision_api_key:
                logger.warning("视觉模型API Key未配置，无法处理图片内容")
                return ""
            
            # 处理图片URL，支持本地文件路径转base64
            processed_images = []
            for url in image_urls[:3]:  # 最多处理3张图片
                if url.startswith(('http://', 'https://')):
                    processed_images.append({
                        "type": "image_url",
                        "image_url": {"url": url}
                    })
                else:
                    # 本地文件路径，转换为base64
                    try:
                        import os
                        if os.path.exists(url):
                            with open(url, "rb") as image_file:
                                image_data = base64.b64encode(image_file.read()).decode()
                                # 根据文件扩展名确定MIME类型
                                ext = url.lower().split('.')[-1]
                                mime_type = f"image/{ext}" if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp'] else "image/jpeg"
                                processed_images.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                                })
                    except Exception as e:
                        logger.warning(f"处理本地图片失败: {e}")
            
            if not processed_images:
                return ""
            
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": "你是一个图片内容识别助手，请客观描述图片内容，特别是提取其中的文字信息。"
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "请描述这张图片的主要内容，特别是如果有文字请完整提取出来。"
                        }
                    ] + processed_images
                }
            ]
            
            # 调用视觉模型
            result = await self._make_openai_request(
                vision_base_url, vision_api_key, vision_model_id, messages, timeout
            )
            
            return result or ""
            
        except Exception as e:
            logger.error(f"图片内容提取失败: {e}")
            return ""
    
    async def _is_spam_message(self, message_content: str, context_messages: List[str], image_content: str = "") -> bool:
        """使用自定义文本模型判断是否为推销消息"""
        try:
            # 获取文本模型配置
            text_model_id = self._get_config_value("TEXT_MODEL_ID", "gpt-3.5-turbo")
            text_base_url = self._get_config_value("TEXT_MODEL_BASE_URL", "https://api.openai.com/v1")
            text_api_key = self._get_config_value("TEXT_MODEL_API_KEY", "")
            timeout = self._get_config_value("MODEL_TIMEOUT", 30)
            
            if not text_api_key:
                logger.warning("文本模型API Key未配置，无法进行推销检测")
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
            
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # 调用文本模型
            result = await self._make_openai_request(
                text_base_url, text_api_key, text_model_id, messages, timeout
            )
            
            if result:
                result = result.strip().lower()
                # 判断模型的回复
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
            
            # 2. 撤回用户最近的所有消息（如果平台支持）
            await self._try_recall_recent_messages(event, user_id, last_time)
            
            # 3. 禁言用户（如果平台支持）
            mute_duration = self._get_config_value("MUTE_DURATION", 600)  # 默认10分钟
            await self._try_mute_user(event, user_id, mute_duration)
            
            # 4. 转发到管理员群
            admin_chat_id = self._get_config_value("ADMIN_CHAT_ID", "")
            if admin_chat_id:
                await self._forward_to_admin(admin_chat_id, user_name, user_id, recent_messages, event)
            
            # 5. 发送警告消息
            alert_message = self._get_config_value("SPAM_ALERT_MESSAGE", 
                "⚠️ 检测到疑似推销信息，该消息已被处理，用户已被禁言。")
            yield event.plain_result(alert_message)
            
        except Exception as e:
            logger.error(f"处理推销消息时出错: {e}")
    
    async def _forward_to_admin(self, admin_chat_id: str, user_name: str, user_id: str, 
                              recent_messages: List[str], event: AstrMessageEvent):
        """转发消息到管理员群"""
        try:
            group_id = event.get_group_id()
            
            # 构建转发内容
            forward_content = f"🚨 推销检测报告\n"
            forward_content += f"用户: {user_name} ({user_id})\n"
            forward_content += f"平台: {event.get_platform_name()}\n"
            forward_content += f"原群聊ID: {group_id or '私聊'}\n"
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
            if event.get_platform_name() == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    payloads = {
                        "message_id": event.message_obj.message_id,
                    }
                    ret = await client.api.call_action('delete_msg', **payloads)
                    logger.info(f"已撤回推销消息: {event.message_obj.message_id}")
        except Exception as e:
            logger.warning(f"撤回消息失败: {e}")
    
    async def _try_recall_recent_messages(self, event: AstrMessageEvent, user_id: str, last_minutes: int):
        """尝试撤回用户最近的所有消息"""
        try:
            if event.get_platform_name() == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    group_id = event.get_group_id()
                    
                    if group_id and user_id in self.user_message_history:
                        cutoff_time = time.time() - (last_minutes * 60)
                        recent_messages = [
                            msg for msg in self.user_message_history[user_id]
                            if msg["timestamp"] > cutoff_time and msg.get("message_id")
                        ]
                        
                        recall_count = 0
                        for msg in recent_messages:
                            try:
                                payloads = {
                                    "message_id": msg["message_id"],
                                }
                                ret = await client.api.call_action('delete_msg', **payloads)
                                recall_count += 1
                                # 避免频繁调用API
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                logger.warning(f"撤回消息 {msg['message_id']} 失败: {e}")
                        
                        if recall_count > 0:
                            logger.info(f"已撤回用户 {user_id} 最近 {recall_count} 条消息")
        except Exception as e:
            logger.warning(f"批量撤回消息失败: {e}")
    
    async def _try_mute_user(self, event: AstrMessageEvent, user_id: str, duration: int):
        """尝试禁言用户（如果平台支持）"""
        try:
            if event.get_platform_name() == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    group_id = event.get_group_id()
                    
                    if group_id:
                        payloads = {
                            "group_id": int(group_id),
                            "user_id": int(user_id),
                            "duration": duration  # 禁言时长（秒）
                        }
                        ret = await client.api.call_action('set_group_ban', **payloads)
                        
                        # 计算禁言时长的可读格式
                        if duration >= 3600:
                            duration_str = f"{duration // 3600}小时{(duration % 3600) // 60}分钟"
                        elif duration >= 60:
                            duration_str = f"{duration // 60}分钟"
                        else:
                            duration_str = f"{duration}秒"
                        
                        logger.info(f"已禁言用户 {user_id}，时长: {duration_str}")
            else:
                logger.warning(f"平台 {event.get_platform_name()} 不支持禁言功能")
        except Exception as e:
            logger.warning(f"禁言用户失败: {e}")
    
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """监听群聊消息"""
        try:
            group_id = event.get_group_id()
            user_id = event.get_sender_id()
            user_name = event.get_sender_name()
            message_content = event.message_str
            timestamp = time.time()
            
            # 群聊白名单检查
            if not self._is_group_whitelisted(group_id):
                logger.debug(f"群聊 {group_id} 不在白名单中，跳过检测")
                return
            
            # 用户白名单检查
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
    
    @filter.command("spam_test", alias={"推销测试"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def test_spam_detection(self, event: AstrMessageEvent, message: str = ""):
        """测试推销检测功能"""
        try:
            if not message:
                yield event.plain_result(
                    "📝 推销检测测试命令使用方法:\n"
                    "/spam_test <消息内容> - 测试指定消息是否为推销信息\n\n"
                    "示例:\n"
                    "/spam_test 优质产品大促销，加微信享受8折优惠！\n"
                    "/spam_test 今天天气真好"
                )
                return
                
            is_spam = await self._is_spam_message(message, [], "")
            result = "✅ 是推销信息" if is_spam else "❌ 不是推销信息"
            yield event.plain_result(f"🔍 推销检测结果: {result}\n测试消息: {message}")
        except Exception as e:
            logger.error(f"测试推销检测时出错: {e}")
            yield event.plain_result("❌ 测试失败，请检查日志和模型配置")
    
    async def terminate(self):
        """插件卸载时的清理工作"""
        logger.info("防推销插件已停止")
        self.user_message_history.clear()
