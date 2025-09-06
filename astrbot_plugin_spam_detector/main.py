import asyncio
import json
import time
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiohttp
from openai import AsyncOpenAI

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
        
    async def _call_text_model(self, messages: List[Dict], model_id: str = None) -> Optional[str]:
        """调用文本模型"""
        try:
            # 获取文本模型配置
            if not model_id:
                model_id = self._get_config_value("TEXT_MODEL_ID", "gpt-3.5-turbo")
            base_url = self._get_config_value("TEXT_MODEL_BASE_URL", "https://api.openai.com/v1")
            api_key = self._get_config_value("TEXT_MODEL_API_KEY", "")
            timeout = self._get_config_value("MODEL_TIMEOUT", 30)
            
            if not api_key:
                logger.warning("文本模型API Key未配置")
                return None
            
            logger.debug(f"调用文本模型: model_id={model_id}, base_url={base_url}, timeout={timeout}")
            
            # 创建OpenAI客户端
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )
            
            # 调用文本模型
            response = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.4,
                max_tokens=1000
            )
            
            if response.choices and len(response.choices) > 0:
                logger.debug(f"文本模型调用成功，返回内容: {response.choices[0].message.content[:100]}...")
                return response.choices[0].message.content
            else:
                logger.warning("文本模型返回空内容")
                
        except Exception as e:
            logger.error(f"文本模型调用失败: {e}", exc_info=True)
        
        return None
    
    async def _call_vision_model(self, messages: List[Dict], model_id: str = None) -> Optional[str]:
        """调用视觉模型"""
        try:
            # 获取视觉模型配置
            if not model_id:
                model_id = self._get_config_value("VISION_MODEL_ID", "gpt-4-vision-preview")
            base_url = self._get_config_value("VISION_MODEL_BASE_URL", "https://api.openai.com/v1")
            api_key = self._get_config_value("VISION_MODEL_API_KEY", "")
            timeout = self._get_config_value("MODEL_TIMEOUT", 30)
            
            if not api_key:
                logger.warning("视觉模型API Key未配置")
                return None
            
            logger.debug(f"���用视觉模型: model_id={model_id}, base_url={base_url}, timeout={timeout}")
            
            # 创建OpenAI客户端
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )
            
            # 调用视觉模型
            response = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.4,
                max_tokens=1000
            )
            
            if response.choices and len(response.choices) > 0:
                logger.debug(f"视觉模型调用成功，返回内容: {response.choices[0].message.content[:100]}...")
                return response.choices[0].message.content
            else:
                logger.warning("视觉模型返回空内容")
                
        except Exception as e:
            logger.error(f"视觉模型调用失败: {e}", exc_info=True)
        
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
            # 检查视觉模型配置
            vision_api_key = self._get_config_value("VISION_MODEL_API_KEY", "")
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
            result = await self._call_vision_model(messages)
            return result or ""
            
        except Exception as e:
            logger.error(f"图片内容提取失败: {e}")
            return ""
    
    async def _is_spam_message(self, message_content: str, context_messages: List[str], image_content: str = "") -> bool:
        """使用自定义文本模型判断是否为推销消息"""
        try:
            # 检查文本模型配置
            text_api_key = self._get_config_value("TEXT_MODEL_API_KEY", "")
            if not text_api_key:
                logger.warning("文本模型API Key未配置，无法进行推销检测")
                return False
            
            logger.debug(f"开始推销检测: message_content={message_content[:50]}..., image_content={image_content[:50]}...")
            
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
            
            logger.debug(f"发送给文本模型的提示词: {prompt[:200]}...")
            
            # 调用文本模型
            result = await self._call_text_model(messages)
            
            if result:
                result = result.strip().lower()
                is_spam = "是" in result or "yes" in result or "spam" in result
                logger.info(f"推销检测模型返回结果: '{result}', 判断为推销: {is_spam}")
                return is_spam
            else:
                logger.warning("推销检测模型未返回结果")
                
        except Exception as e:
            logger.error(f"推销检测失败: {e}", exc_info=True)
        
        return False
    
    async def _handle_spam_message(self, event: AstrMessageEvent, user_id: str, user_name: str):
        """处理检测到的推销消息"""
        try:
            logger.info(f"开始处理推销消息，用户: {user_name} ({user_id})")
            
            # 1. 获取用户最近的消息
            last_time = self._get_config_value("LAST_TIME", 5)
            recent_messages = self._get_user_recent_messages(user_id, last_time)
            logger.info(f"获取到用户 {user_id} 最近 {last_time} 分钟内的 {len(recent_messages)} 条消息")
            
            # 2. 撤回用户最近的所有消息（如果平台支持）
            logger.info("步骤2: 尝试撤回消息")
            await self._try_recall_recent_messages(event, user_id, last_time)
            
            # 3. 禁言用户（如果平台支持）
            mute_duration = self._get_config_value("MUTE_DURATION", 600)  # 默认10分钟
            logger.info(f"步骤3: 尝试禁言用户 {mute_duration} 秒")
            await self._try_mute_user(event, user_id, mute_duration)
            
            # 4. 转发到管理员群
            admin_chat_id = self._get_config_value("ADMIN_CHAT_ID", "")
            if admin_chat_id:
                logger.info(f"步骤4: 转发推销消息到管理员群: {admin_chat_id}")
                await self._forward_to_admin(admin_chat_id, user_name, user_id, recent_messages, event)
            else:
                logger.warning("步骤4: 管理员群聊ID未配置，无法转发推销消息")
            
            # 5. 发送警告消息
            alert_message = self._get_config_value("SPAM_ALERT_MESSAGE",
                "⚠️ 检测到疑似推销信息，该消息已被处理，用户已被禁言。")
            logger.info(f"步骤5: 发送警告消息: {alert_message}")
            yield event.plain_result(alert_message)
            
        except Exception as e:
            logger.error(f"处理推销消息时出错: {e}", exc_info=True)
            yield event.plain_result("❌ 处理推销消息时发生错误，请检查日志")
    
    async def _forward_to_admin(self, admin_chat_id: str, user_name: str, user_id: str, 
                              recent_messages: List[str], event: AstrMessageEvent):
        """转发消息到管理员群"""
        try:
            if not admin_chat_id:
                logger.warning("管理员群聊ID未配置，无法转发消息")
                return
                
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
            
            # 使用正确的MessageChain导入和发送方式
            from astrbot.api.event import MessageChain
            message_chain = MessageChain().message(forward_content)
            
            logger.info(f"正在转发推销报告到管理员群: {admin_unified_origin}")
            await self.context.send_message(admin_unified_origin, message_chain)
            logger.info("推销报告转发成功")
            
        except Exception as e:
            logger.error(f"转发到管理员群失败: {e}", exc_info=True)
    
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
            platform_name = event.get_platform_name()
            logger.info(f"尝试撤回用户 {user_id} 最近 {last_minutes} 分钟的消息，平台: {platform_name}")
            
            if platform_name == "aiocqhttp":
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
                        
                        logger.info(f"找到用户 {user_id} 需要撤回的消息: {len(recent_messages)} 条")
                        
                        recall_count = 0
                        for msg in recent_messages:
                            try:
                                logger.debug(f"正在撤回消息ID: {msg['message_id']}")
                                payloads = {
                                    "message_id": msg["message_id"],
                                }
                                ret = await client.api.call_action('delete_msg', **payloads)
                                recall_count += 1
                                logger.debug(f"成功撤回消息 {msg['message_id']}: {msg['content'][:30]}...")
                                # 避免频繁调用API
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                logger.warning(f"撤回消息 {msg['message_id']} 失败: {e}")
                        
                        if recall_count > 0:
                            logger.info(f"✅ 已撤回用户 {user_id} 最近 {recall_count} 条消息")
                        else:
                            logger.warning(f"未能撤回用户 {user_id} 的任何消息")
                    else:
                        logger.warning(f"无法撤回消息: 群聊ID或用户历史消息不存在")
            else:
                logger.warning(f"平台 {platform_name} 不支持消息撤回功能")
        except Exception as e:
            logger.error(f"批量撤回消息失败: {e}", exc_info=True)
    
    async def _try_mute_user(self, event: AstrMessageEvent, user_id: str, duration: int):
        """尝试禁言用户（如果平台支持）"""
        try:
            platform_name = event.get_platform_name()
            logger.info(f"尝试禁言用户 {user_id}，时长: {duration}秒，平台: {platform_name}")
            
            if platform_name == "aiocqhttp":
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
                        logger.debug(f"调用 set_group_ban API，payloads: {payloads}")
                        ret = await client.api.call_action('set_group_ban', **payloads)
                        
                        # 计算禁言时长的可读格式
                        if duration >= 3600:
                            duration_str = f"{duration // 3600}小时{(duration % 3600) // 60}分钟"
                        elif duration >= 60:
                            duration_str = f"{duration // 60}分钟"
                        else:
                            duration_str = f"{duration}秒"
                        
                        logger.info(f"✅ 已禁言用户 {user_id}，时长: {duration_str}")
                    else:
                        logger.warning(f"无法禁言用户 {user_id}: 群聊ID不存在")
            else:
                logger.warning(f"平台 {platform_name} 不支持禁言功能")
        except Exception as e:
            logger.warning(f"禁言用户失败: {e}", exc_info=True)
    
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """监听群聊消息"""
        try:
            group_id = event.get_group_id()
            user_id = event.get_sender_id()
            user_name = event.get_sender_name()
            message_content = event.message_str
            timestamp = time.time()
            
            logger.debug(f"收到群聊消息: 群聊 {group_id}, 用户 {user_id}, 内容: {message_content[:50]}...")
            
            # 群聊白名单检查
            if not self._is_group_whitelisted(group_id):
                logger.debug(f"群聊 {group_id} 不在白名单中，跳过检测")
                return
            logger.debug(f"群聊 {group_id} 在白名单中")
            
            # 用户白名单检查
            if self._is_user_whitelisted(user_id):
                logger.debug(f"用户 {user_id} 在白名单中，跳过检测")
                return
            logger.debug(f"用户 {user_id} 不在白名单中")
            
            # 存储用户消息
            message_id = getattr(event.message_obj, 'message_id', '')
            self._store_user_message(user_id, message_content, timestamp, message_id)
            logger.debug(f"已存储用户 {user_id} 的消息，消息ID: {message_id}")
            
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
                logger.debug(f"检测到图片: {len(image_urls)} 张")
                image_content = await self._extract_image_content(image_urls)
                if image_content:
                    logger.debug(f"图片内容提取成功: {image_content[:100]}...")
                else:
                    logger.debug("图片内容提取失败或无内容")
            
            # 获取上下文消息
            context_count = self._get_config_value("CONTEXT_MESSAGE_COUNT", 1)
            context_messages = await self._get_context_messages(event, context_count)
            logger.debug(f"获取到 {len(context_messages)} 条上下文消息")
            
            # 检测是否为推销消息
            logger.debug(f"开始检测推销消息: {message_content[:50]}...")
            is_spam = await self._is_spam_message(message_content, context_messages, image_content)
            
            if is_spam:
                logger.info(f"🚨 检测到推销消息，用户: {user_name} ({user_id}), 内容: {message_content}")
                async for result in self._handle_spam_message(event, user_id, user_name):
                    yield result
            else:
                logger.debug(f"消息检测结果: 非推销消息")
                
        except Exception as e:
            logger.error(f"处理群聊消息时出错: {e}", exc_info=True)
    
    @filter.command("spam_test", alias={"推销测试"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def test_spam_detection(self, event: AstrMessageEvent, message: str = ""):
        """测试推销检测功能"""
        try:
            # 将所有参数合并为一个消息字符串
            # message = " ".join(args) if args else ""
            
            if not message:
                yield event.plain_result(
                    "📝 推销检测测试命令使用方法:\n"
                    "/spam_test <消息内容> - 测试指定消息是否为推销信息\n\n"
                    "示例:\n"
                    "/spam_test 优质产品大促销，加微信享受8折优惠！\n"
                    "/spam_test 今天天气真好"
                )
                return
                
            logger.info(f"开始测试推销检测: {message}")
            is_spam = await self._is_spam_message(message, [], "")
            result = "✅ 是推销信息" if is_spam else "❌ 不是推销信息"
            yield event.plain_result(f"🔍 推销检测结果: {result}\n测试消息: {message}")
        except Exception as e:
            logger.error(f"测试推销检测时出错: {e}", exc_info=True)
            yield event.plain_result("❌ 测试失败，请检查日志和模型配置")
    
    @filter.command("spam_debug", alias={"推销调试"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def debug_spam_plugin(self, event: AstrMessageEvent):
        """调试推销插件状态"""
        try:
            config_status = []
            
            # 检查配置项
            admin_chat_id = self._get_config_value("ADMIN_CHAT_ID", "")
            config_status.append(f"管理员群聊ID: {'已配置' if admin_chat_id else '❌ 未配置'} ({admin_chat_id})")
            
            group_whitelist = self._get_config_value("GROUP_WHITELIST", [])
            config_status.append(f"群聊白名单: {len(group_whitelist)} 个群聊")
            
            user_whitelist = self._get_config_value("WHITELIST_USERS", [])
            config_status.append(f"用户白名单: {len(user_whitelist)} 个用户")
            
            # 检查模型配置
            text_model_api_key = self._get_config_value("TEXT_MODEL_API_KEY", "")
            config_status.append(f"文本模型API Key: {'已配置' if text_model_api_key else '❌ 未配置'}")
            
            vision_model_api_key = self._get_config_value("VISION_MODEL_API_KEY", "")
            config_status.append(f"视觉模型API Key: {'已配置' if vision_model_api_key else '❌ 未配置'}")
            
            # 检查当前群聊状态
            current_group = event.get_group_id()
            if current_group:
                is_group_whitelisted = self._is_group_whitelisted(current_group)
                config_status.append(f"当前群聊 {current_group}: {'✅ 在白名单中' if is_group_whitelisted else '❌ 不在白名单中'}")
            
            # 检查消息历史
            total_cached_users = len(self.user_message_history)
            config_status.append(f"缓存用户消息: {total_cached_users} 个用户")
            
            debug_info = "🔧 推销插件调试信息:\n" + "\n".join(f"• {status}" for status in config_status)
            yield event.plain_result(debug_info)
            
        except Exception as e:
            logger.error(f"调试插件状态时出错: {e}", exc_info=True)
            yield event.plain_result("❌ 调试失败，请检查日志")
    
    @filter.command("spam_test_forward", alias={"测试转发"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def test_forward_function(self, event: AstrMessageEvent):
        """测试转发功能"""
        try:
            admin_chat_id = self._get_config_value("ADMIN_CHAT_ID", "")
            if not admin_chat_id:
                yield event.plain_result("❌ 管理员群聊ID未配置，无法测试转发功能")
                return
            
            # 模拟推销消息数据
            test_user_id = event.get_sender_id()
            test_user_name = event.get_sender_name()
            test_messages = ["这是测试消息1", "这是测试消息2"]
            
            logger.info(f"开始测试转发功能到群聊: {admin_chat_id}")
            await self._forward_to_admin(admin_chat_id, test_user_name, test_user_id, test_messages, event)
            
            yield event.plain_result(f"✅ 转发测试完成，已发送到群聊: {admin_chat_id}")
            
        except Exception as e:
            logger.error(f"测试转发功能时出错: {e}", exc_info=True)
            yield event.plain_result("❌ 转发测试失败，请检查日志和配置")
    
    async def terminate(self):
        """插件卸载时的清理工作"""
        logger.info("防推销插件已停止")
        self.user_message_history.clear()
