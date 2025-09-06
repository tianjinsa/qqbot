import asyncio
import time
import base64
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig
import astrbot.api.message_components as Comp


@register("astrbot_plugin_spam_detector", "AstrBot Dev Team", "智能防推销插件，使用AI检测并处理推销信息", "1.1.2", "https://github.com/AstrBotDevs/astrbot_plugin_spam_detector")
class SpamDetectorPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 每个群聊的消息池：群聊ID -> {用户ID -> [消息记录]}
        self.group_message_pools = {}  # type: Dict[str, Dict[str, List[Dict[str, Any]]]]
        # 推销检测处理队列：存储(群聊ID, 用户ID, 消息内容, 发送时间, 事件对象)
        self.detection_queue = asyncio.Queue()
        self.last_model_call_time = 0.0
        self.detection_worker_running = False
        # 批量处理缓冲区：群聊ID -> [检测任务列表]
        self.batch_buffer = {}  # type: Dict[str, List[tuple]]
        self.batch_timer = {}  # type: Dict[str, float]
        
    async def initialize(self):
        """插件初始化"""
        logger.info("防推销插件已启动")
        # 启动检测队列处理器
        asyncio.create_task(self._detection_worker())
        
    async def _detection_worker(self):
        """队列处理器：支持批量处理和速率限制的推销检测请求"""
        self.detection_worker_running = True
        logger.info("推销检测队列处理器已启动")
        
        while self.detection_worker_running:
            try:
                # 获取配置
                batch_size = int(self._get_config_value("BATCH_PROCESS_SIZE", 3))
                rate_limit = float(self._get_config_value("QUEUE_RATE_LIMIT", 1.0))
                
                # 等待队列中的检测任务
                detection_task = await self.detection_queue.get()
                group_id, user_id, user_name, message_content, timestamp, event, image_content = detection_task
                
                # 初始化群聊的批量缓冲区
                if group_id not in self.batch_buffer:
                    self.batch_buffer[group_id] = []
                    self.batch_timer[group_id] = time.time()
                
                # 添加任务到批量缓冲区
                self.batch_buffer[group_id].append(detection_task)
                
                # 检查是否需要批量处理
                should_process = (
                    len(self.batch_buffer[group_id]) >= batch_size or  # 达到批量大小
                    time.time() - self.batch_timer[group_id] > 5.0  # 超过5秒等待时间
                )
                
                if should_process:
                    # 速率限制：确保距离上次调用至少rate_limit秒
                    now = time.time()
                    time_since_last_call = now - self.last_model_call_time
                    if time_since_last_call < rate_limit:
                        await asyncio.sleep(rate_limit - time_since_last_call)
                    
                    # 更新最后调用时间
                    self.last_model_call_time = time.time()
                    
                    # 处理批量任务
                    tasks_to_process = self.batch_buffer[group_id].copy()
                    self.batch_buffer[group_id].clear()
                    self.batch_timer[group_id] = time.time()
                    
                    await self._process_batch_tasks(group_id, tasks_to_process)
                
                # 标记任务完成
                self.detection_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("推销检测队列处理器被取消")
                break
            except Exception as e:
                logger.error(f"队列处理器出错: {e}", exc_info=True)
                await asyncio.sleep(1)  # 出错时暂停1秒
        
        logger.info("推销检测队列处理器已停止")
    
    async def _process_batch_tasks(self, group_id: str, tasks: List[tuple]):
        """批量处理同一群聊的检测任务"""
        try:
            # 获取批量处理配置
            max_batch_text_length = int(self._get_config_value("BATCH_MAX_TEXT_LENGTH", 2000))
            batch_size = int(self._get_config_value("BATCH_PROCESS_SIZE", 3))
            
            logger.info(f"开始批量处理群聊 {group_id} 的 {len(tasks)} 条消息")
            
            # 第一批：符合字数和数量限制的消息
            main_batch_tasks = []
            remaining_tasks = []
            total_text_length = 0
            
            for task in tasks:
                if len(main_batch_tasks) >= batch_size:
                    remaining_tasks.extend(tasks[len(main_batch_tasks):])
                    break
                    
                full_content = self._build_full_content(task)
                
                # 检查是否超过字数限制
                if total_text_length + len(full_content) > max_batch_text_length:
                    # 如果第一条消息就超过限制，单独处理
                    if len(main_batch_tasks) == 0:
                        await self._process_single_task(task, group_id, "消息过长，单独处理")
                        remaining_tasks.extend(tasks[1:])
                    else:
                        remaining_tasks.extend(tasks[len(main_batch_tasks):])
                    break
                
                main_batch_tasks.append(task)
                total_text_length += len(full_content)
            
            # 处理主批量
            if main_batch_tasks:
                await self._process_task_batch(main_batch_tasks, group_id, "主批量")
            
            # 处理剩余任务
            if remaining_tasks:
                await self._process_task_batch(remaining_tasks, group_id, "剩余任务")
                
        except Exception as e:
            logger.error(f"批量处理任务时出错: {e}", exc_info=True)
            # 错误回退：逐条处理
            logger.info(f"尝试逐条批量处理 {len(tasks)} 条消息")
            for task in tasks:
                try:
                    await self._process_single_task(task, group_id, "回退处理")
                except Exception as single_e:
                    logger.error(f"单个任务处理失败: {single_e}", exc_info=True)
    
    def _build_full_content(self, task: tuple) -> str:
        """构建完整的消息内容（文本+图片）"""
        _, user_id, user_name, message_content, timestamp, event, image_content = task
        full_content = message_content
        if image_content:
            full_content += f"\n图片内容：{image_content}"
        return full_content
    
    def _extract_task_info(self, task: tuple) -> tuple:
        """提取任务信息"""
        _, user_id, user_name, message_content, timestamp, event, image_content = task
        return user_id, user_name, message_content, timestamp, event, image_content
    
    async def _process_task_batch(self, tasks: List[tuple], group_id: str, batch_type: str):
        """处理一批任务"""
        if not tasks:
            return
            
        logger.info(f"开始{batch_type}处理 {len(tasks)} 条消息")
        
        # 构建批量输入
        batch_input = {}
        task_map = {}
        
        for task in tasks:
            user_id, user_name, message_content, timestamp, event, image_content = self._extract_task_info(task)
            full_content = self._build_full_content(task)
            batch_input[user_id] = full_content
            task_map[user_id] = task
        
        # 批量检测
        total_chars = sum(len(content) for content in batch_input.values())
        logger.info(f"{batch_type}检测 {len(batch_input)} 条消息，总字符数: {total_chars}")
        
        spam_user_ids = await self._batch_spam_detection(batch_input)
        
        # 处理检测结果
        for user_id in spam_user_ids:
            if user_id in task_map:
                task = task_map[user_id]
                user_id, user_name, message_content, timestamp, event, image_content = self._extract_task_info(task)
                await self._handle_spam_detection_result(user_id, user_name, group_id, event, batch_type)
        
        logger.info(f"{batch_type}处理完成，发现 {len(spam_user_ids)} 个推销用户，分别是: {', '.join(spam_user_ids)}")
    
    async def _process_single_task(self, task: tuple, group_id: str, reason: str):
        """处理单个任务"""
        user_id, user_name, message_content, timestamp, event, image_content = self._extract_task_info(task)
        full_content = self._build_full_content(task)
        
        logger.warning(f"{reason}: {full_content[:50]}... (用户: {user_name})")
        
        # 使用单条批量检测
        single_batch_input = {user_id: full_content}
        spam_user_ids = await self._batch_spam_detection(single_batch_input)
        
        if user_id in spam_user_ids:
            await self._handle_spam_detection_result(user_id, user_name, group_id, event, reason)
    
    async def _handle_spam_detection_result(self, user_id: str, user_name: str, group_id: str, event, context: str):
        """处理推销检测结果"""
        logger.info(f"🚨 {context}检测到推销消息，用户: {user_name} ({user_id}), 群聊: {group_id}")
        await self._handle_spam_message_new(event, group_id, user_id, user_name)
    
    async def _batch_spam_detection(self, batch_input: Dict[str, str]) -> List[str]:
        """批量推销检测，返回被识别为推销的用户ID列表"""
        try:
            # 检查文本模型配置
            text_api_key = self._get_config_value("TEXT_MODEL_API_KEY", "")
            if not text_api_key:
                logger.warning("文本模型API Key未配置，无法进行批量推销检测")
                return []
            
            logger.debug(f"开始批量推销检测，消息数量: {len(batch_input)}")
            
            # 构建批量检测的提示词
            batch_content = json.dumps(batch_input, ensure_ascii=False, indent=2)
            
            # 获取系统提示词
            system_prompt = self._get_config_value("LLM_SYSTEM_PROMPT",
                """你是一个专业的推销信息检测助手。你将收到一个JSON格式的批量消息，其中包含多个用户的消息内容。

推销信息的特征包括但不限于：
1. 销售产品或服务
2. 包含价格、优惠、折扣等商业信息
3. 引导添加微信、QQ等联系方式进行交易
4. 推广某个商品、品牌或服务
5. 含有明显的营销意图

请分析所有消息，找出其中的推销信息，并返回一个JSON格式的结果，格式为：{"y":[用户ID1,用户ID2,...]}
其中y数组包含所有被识别为推销信息的用户ID。如果没有推销信息，返回{"y":[]}""")
            
            prompt = f"请分析以下批量消息，识别出推销信息的用户ID：\n\n{batch_content}"
            
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            logger.debug(f"发送给文本模型的批量检测提示词: {prompt[:200]}...")
            
            # 调用文本模型进行批量检测
            result = await self._call_text_model(messages)
            logger.info(f"模型返回: {result}")
            if result:
                try:
                    # 解析JSON结果
                    result_json = json.loads(result.strip())
                    spam_user_ids = result_json.get("y", [])
                    
                    # 确保返回的是字符串列表
                    spam_user_ids = [str(uid) for uid in spam_user_ids]
                    
                    logger.info(f"批量推销检测模型返回结果: {spam_user_ids}")
                    return spam_user_ids
                except json.JSONDecodeError as e:
                    logger.warning(f"批量检测结果JSON解析失败: {e}, 原始结果: {result}")
                    return []
            else:
                logger.warning("批量推销检测模型未返回结果")
                return []
                
        except Exception as e:
            logger.error(f"批量推销检测失败: {e}", exc_info=True)
            return []
        
    async def _call_text_model(self, messages: List[Dict], model_id: str = None) -> Optional[str]:
        """调用文本模型"""
        try:
            # 获取文本模型配置
            if not model_id:
                model_id = self._get_config_value("TEXT_MODEL_ID", "gpt-3.5-turbo")
            base_url = self._get_config_value("TEXT_MODEL_BASE_URL", "https://api.openai.com/v1")
            api_key = self._get_config_value("TEXT_MODEL_API_KEY", "")
            timeout = self._get_config_value("MODEL_TIMEOUT", 30)
            temperature = self._get_config_value("TEXT_MODEL_TEMPERATURE", 0.7)
            thinking_enabled = self._get_config_value("TEXT_MODEL_THINKING_ENABLED", False)
            
            if not api_key:
                logger.warning("文本模型API Key未配置")
                return None
            
            logger.debug(f"调用文本模型: model_id={model_id}, base_url={base_url}, timeout={timeout}, temperature={temperature}, thinking_enabled={thinking_enabled}")
            
            # 调试信息：打印即将发送的消息
            logger.debug(f"发送给模型的消息数量: {len(messages)}")
            for i, msg in enumerate(messages):
                logger.debug(f"消息 {i+1}: role={msg.get('role')}, content长度={len(str(msg.get('content', '')))}")
            
            # 创建OpenAI客户端
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )
            
            # 构建基础API调用参数
            api_params = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1000
            }
            
            logger.debug(f"最终API参数: {', '.join(api_params.keys())}")
            
            # 调用文本模型
            if thinking_enabled:
                # 只使用extra_body方式传递thinking参数
                try:
                    logger.debug("使用extra_body方式启用thinking模式")
                    response = await client.chat.completions.create(
                        **api_params,
                        extra_body={"thinking": {"type": "enabled"}}
                    )
                    logger.debug("成功使用extra_body方式启用thinking模式")
                except Exception as e:
                    logger.warning(f"extra_body thinking模式失败，回退到普通模式: {e}")
                    response = await client.chat.completions.create(**api_params)
            else:
                response = await client.chat.completions.create(**api_params)
            
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
            temperature = self._get_config_value("VISION_MODEL_TEMPERATURE", 0.7)
            thinking_enabled = self._get_config_value("VISION_MODEL_THINKING_ENABLED", False)
            system_prompt = self._get_config_value("VISION_MODEL_SYSTEM_PROMPT", "提取图片上的内容，特别是文字")
            
            if not api_key:
                logger.warning("视觉模型API Key未配置")
                return None
            
            logger.debug(f"调用视觉模型: model_id={model_id}, base_url={base_url}, timeout={timeout}, temperature={temperature}, thinking_enabled={thinking_enabled}")
            
            # 调试信息：打印即将发送的消息
            logger.debug(f"发送给视觉模型的消息数量: {len(messages)}")
            for i, msg in enumerate(messages):
                logger.debug(f"消息 {i+1}: role={msg.get('role')}, content类型={type(msg.get('content'))}")
            
            # 创建OpenAI客户端
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )
            
            # 确保有系统消息
            final_messages = []
            has_system = any(msg.get('role') == 'system' for msg in messages)
            if not has_system:
                final_messages.append({"role": "system", "content": system_prompt})
            final_messages.extend(messages)
            
            # 构建基础API调用参数
            api_params = {
                "model": model_id,
                "messages": final_messages,
                "temperature": temperature,
                "max_tokens": 1000
            }
            
            logger.debug(f"最终API参数: {', '.join(api_params.keys())}")
            
            # 调用视觉模型
            if thinking_enabled:
                # 只使用extra_body方式传递thinking参数
                try:
                    logger.debug("使用extra_body方式启用thinking模式")
                    response = await client.chat.completions.create(
                        **api_params,
                        extra_body={"thinking": {"type": "enabled"}}
                    )
                    logger.debug("成功使用extra_body方式启用thinking模式")
                except Exception as e:
                    logger.warning(f"extra_body thinking模式失败，回退到普通模式: {e}")
                    response = await client.chat.completions.create(**api_params)
            else:
                response = await client.chat.completions.create(**api_params)
            
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
    
    def _add_message_to_pool(self, group_id: str, user_id: str, message_content: str, 
                            timestamp: float, message_id: str = ""):
        """将消息添加到对应群聊的消息池中"""
        if group_id not in self.group_message_pools:
            self.group_message_pools[group_id] = {}
        
        if user_id not in self.group_message_pools[group_id]:
            self.group_message_pools[group_id][user_id] = []
        
        # 添加消息记录
        message_record = {
            "content": message_content,
            "timestamp": timestamp,
            "message_id": message_id,
            "recalled": False
        }
        
        self.group_message_pools[group_id][user_id].append(message_record)
        
        # 清理过期消息
        self._cleanup_expired_messages(group_id, timestamp)
        
        logger.debug(f"已添加消息到群聊 {group_id} 用户 {user_id} 的消息池，当前池大小: {len(self.group_message_pools[group_id][user_id])}")
    
    def _cleanup_expired_messages(self, group_id: str, current_timestamp: float):
        """清理指定群聊中超过LAST_TIME的过期消息"""
        if group_id not in self.group_message_pools:
            return
        
        last_time_minutes = int(self._get_config_value("LAST_TIME", 5))
        cutoff_time = current_timestamp - (last_time_minutes * 60)
        
        # 清理每个用户的过期消息
        users_to_remove = []
        for user_id, messages in self.group_message_pools[group_id].items():
            # 保留未过期的消息
            valid_messages = [msg for msg in messages if msg["timestamp"] > cutoff_time]
            
            if valid_messages:
                self.group_message_pools[group_id][user_id] = valid_messages
            else:
                users_to_remove.append(user_id)
        
        # 移除没有消息的用户
        for user_id in users_to_remove:
            del self.group_message_pools[group_id][user_id]
        
        # 如果群聊中没有任何用户消息，移除整个群聊记录
        if not self.group_message_pools[group_id]:
            del self.group_message_pools[group_id]
    
    def _get_user_messages_in_group(self, group_id: str, user_id: str) -> List[Dict[str, Any]]:
        """获取指定群聊中指定用户的所有消息"""
        if group_id not in self.group_message_pools:
            return []
        
        if user_id not in self.group_message_pools[group_id]:
            return []
        
        return self.group_message_pools[group_id][user_id].copy()
    
    def _remove_recalled_message(self, group_id: str, user_id: str, message_id: str):
        """从消息池中删除已撤回的消息"""
        if group_id not in self.group_message_pools:
            return
        
        if user_id not in self.group_message_pools[group_id]:
            return
        
        # 标记消息为已撤回并从列表中移除
        messages = self.group_message_pools[group_id][user_id]
        for i, msg in enumerate(messages):
            if msg.get("message_id") == message_id:
                messages.pop(i)
                logger.debug(f"已从消息池中删除撤回的消息: {message_id}")
                break
    
    def _clear_user_detection_queue(self, group_id: str, user_id: str):
        """从检测队列中清理指定群聊指定用户的待处理任务"""
        try:
            # 创建临时队列存储不需要清理的任务
            temp_queue = asyncio.Queue()
            cleared_count = 0
            
            # 从原队列中取出所有任务
            while not self.detection_queue.empty():
                try:
                    task = self.detection_queue.get_nowait()
                    task_group_id, task_user_id = task[0], task[1]
                    
                    # 如果不是要清理的用户任务，放入临时队列
                    if task_group_id != group_id or task_user_id != user_id:
                        temp_queue.put_nowait(task)
                    else:
                        cleared_count += 1
                        logger.debug(f"从队列中清除任务: 群聊{task_group_id}, 用户{task_user_id}")
                        
                except asyncio.QueueEmpty:
                    break
            
            # 将临时队列中的任务放回原队列
            while not temp_queue.empty():
                try:
                    task = temp_queue.get_nowait()
                    self.detection_queue.put_nowait(task)
                except asyncio.QueueEmpty:
                    break
            
            if cleared_count > 0:
                logger.info(f"已从检测队列中清除 {cleared_count} 个重复任务 (群聊: {group_id}, 用户: {user_id})")
                
        except Exception as e:
            logger.error(f"清理检测队列时出错: {e}", exc_info=True)
    
    async def _forward_messages_as_merged(self, admin_chat_id: str, group_id: str, user_id: str, 
                                        user_name: str, user_messages: List[Dict], event: AstrMessageEvent):
        """使用合并转发的方式将消息转发到管理员群"""
        try:
            if not admin_chat_id:
                logger.warning("管理员群聊ID未配置，无法转发消息")
                return
                
            platform_name = event.get_platform_name()
            if platform_name != "aiocqhttp":
                logger.warning(f"平台 {platform_name} 不支持合并转发，使用文本转发")
                await self._forward_to_admin_text(admin_chat_id, group_id, user_id, user_name, user_messages, event)
                return
            
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
            if not isinstance(event, AiocqhttpMessageEvent):
                logger.warning("事件类型不是 AiocqhttpMessageEvent，无法使用合并转发")
                await self._forward_to_admin_text(admin_chat_id, group_id, user_id, user_name, user_messages, event)
                return
            
            client = event.bot
            group_name = await self._get_group_name(group_id)
            
            # 构建合并转发的节点列表
            import astrbot.api.message_components as Comp
            nodes = []
            
            # 添加标题节点
            title_content = f"🚨 推销检测报告\n👤 用户: {user_name} ({user_id})\n🏷️ 原群聊: {group_name} ({group_id})\n⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            nodes.append(Comp.Node(
                uin=str(client.self_id),
                name="AstrBot反推销系统",
                content=[Comp.Plain(title_content)]
            ))
            
            # 添加每条被撤回的消息作为节点
            for i, msg_record in enumerate(user_messages):
                if msg_record.get("content", "").strip():
                    timestamp_str = datetime.fromtimestamp(msg_record.get("timestamp", time.time())).strftime('%H:%M:%S')
                    content_text = f"[{timestamp_str}] {msg_record['content']}"
                    nodes.append(Comp.Node(
                        uin=str(user_id),
                        name=f"{user_name}",
                        content=[Comp.Plain(content_text)]
                    ))
            
            if len(nodes) <= 1:
                logger.warning("没有有效的消息内容，跳过合并转发")
                return
            
            # 发送合并转发
            logger.info(f"发送合并转发到管理员群 {admin_chat_id}，包含 {len(nodes)} 个节点")
            
            # 使用原生 CQHTTP API 发送合并转发
            forward_msg = []
            for node in nodes:
                forward_msg.append({
                    "type": "node",
                    "data": {
                        "uin": str(node.uin),
                        "name": node.name,
                        "content": [{"type": "text", "data": {"text": comp.text}} for comp in node.content if hasattr(comp, 'text')]
                    }
                })
            
            ret = await client.api.call_action(
                'send_group_forward_msg',
                group_id=str(admin_chat_id),
                messages=forward_msg
            )
            logger.info(f"合并转发结果: {ret}")
            
        except Exception as e:
            logger.error(f"合并转发失败: {e}", exc_info=True)
            # 回退到文本转发
            await self._forward_to_admin_text(admin_chat_id, group_id, user_id, user_name, user_messages, event)
    
    async def _forward_to_admin_text(self, admin_chat_id: str, group_id: str, user_id: str,
                                   user_name: str, user_messages: List[Dict], event: AstrMessageEvent):
        """文本形式转发到管理员群（作为合并转发的备用方案）"""
        try:
            group_name = await self._get_group_name(group_id)
            
            # 构建转发内容
            forward_content = f"🚨 推销检测报告\n"
            forward_content += f"👤 用户: {user_name} ({user_id})\n"
            forward_content += f"🏷️ 原群聊: {group_name} ({group_id})\n"
            forward_content += f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            forward_content += f"📋 被撤回的消息 ({len(user_messages)} 条):\n"
            
            for i, msg_record in enumerate(user_messages, 1):
                if msg_record.get("content", "").strip():
                    timestamp_str = datetime.fromtimestamp(msg_record.get("timestamp", time.time())).strftime('%H:%M:%S')
                    forward_content += f"{i}. [{timestamp_str}] {msg_record['content']}\n"
            
            platform_name = event.get_platform_name()
            if platform_name == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    ret = await client.api.call_action(
                        'send_group_msg',
                        group_id=str(admin_chat_id),
                        message=forward_content
                    )
                    logger.info(f"文本转发结果: {ret}")
            
        except Exception as e:
            logger.error(f"文本转发失败: {e}", exc_info=True)
    
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
            
            # 处理图片URL，支持HTTP链接和本地文件路径转base64
            processed_images = []
            for i, url in enumerate(image_urls[:4]):  # 最多处理4张图片
                logger.debug(f"处理图片 {i+1}/{len(image_urls[:4])}: {url}")
                
                if url.startswith(('http://', 'https://')):
                    # HTTP/HTTPS链接，直接使用URL
                    processed_images.append({
                        "type": "image_url",
                        "image_url": {"url": url}
                    })
                    logger.debug(f"图片 {i+1}: 使用HTTP链接格式")
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
                                logger.debug(f"图片 {i+1}: 成功转换为base64格式 ({mime_type})")
                        else:
                            logger.warning(f"本地文件不存在: {url}")
                    except Exception as e:
                        logger.warning(f"处理本地图片失败: {e}")
            
            if not processed_images:
                return ""
            
            # 构建符合GLM-4.1v格式的消息
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
            
            logger.debug(f"发送给视觉模型的消息格式:")
            logger.debug(f"- 系统消息: {messages[0]['content']}")
            logger.debug(f"- 用户消息包含 {len(processed_images)} 张图片")
            for i, img in enumerate(processed_images):
                img_url = img["image_url"]["url"]
                if img_url.startswith("data:"):
                    logger.debug(f"  图片{i+1}: base64格式 (长度: {len(img_url)} 字符)")
                elif img_url.startswith(('http://', 'https://')):
                    logger.debug(f"  图片{i+1}: HTTP链接格式 ({img_url})")
                else:
                    logger.debug(f"  图片{i+1}: 其他格式 ({img_url[:50]}...)")
            
            # 调用视觉模型
            result = await self._call_vision_model(messages)
            return result or ""
            
        except Exception as e:
            logger.error(f"图片内容提取失败: {e}")
            return ""
    
    async def _handle_spam_message_new(self, event: AstrMessageEvent, group_id: str, user_id: str, user_name: str) -> Optional[Comp.BaseMessageComponent]:
        """处理检测到的推销消息 - 新的逻辑流程"""
        try:
            logger.info(f"开始处理推销消息，用户: {user_name} ({user_id})，群聊: {group_id}")
            
            # 0. 清理检测队列中同一群聊同一用户的重复任务
            logger.info(f"步骤0: 清理检测队列中的重复任务")
            self._clear_user_detection_queue(group_id, user_id)
            
            # 1. 先禁言用户
            mute_duration = self._get_config_value("MUTE_DURATION", 600)  # 默认10分钟
            logger.info(f"步骤1: 禁言用户 {user_id}，时长: {mute_duration} 秒")
            await self._try_mute_user(event, user_id, mute_duration)
            
            # 2. 从消息池中获取该用户的所有消息
            user_messages = self._get_user_messages_in_group(group_id, user_id)
            logger.info(f"步骤2: 从消息池获取到用户 {user_id} 的 {len(user_messages)} 条消息")
            
            # 3. 先进行合并转发到管理员群（在撤回之前）
            admin_chat_id = self._get_config_value("ADMIN_CHAT_ID", "")
            if admin_chat_id and user_messages:
                logger.info(f"步骤3: 合并转发推销消息到管理员群: {admin_chat_id}")
                await self._forward_messages_as_merged(admin_chat_id, group_id, user_id, user_name, user_messages, event)
            elif not admin_chat_id:
                logger.warning("步骤3: 管理员群聊ID未配置，跳过转发")
            else:
                logger.warning("步骤3: 没有消息可转发")
            
            # 4. 执行消息撤回
            logger.info(f"步骤4: 开始撤回用户 {user_id} 的消息")
            recall_count = 0
            for message_record in user_messages:
                message_id = message_record.get("message_id")
                if message_id and not message_record.get("recalled"):
                    try:
                        success = await self._try_recall_message_by_id(event, message_id)
                        if success:
                            # 从消息池中删除撤回的消息
                            self._remove_recalled_message(group_id, user_id, message_id)
                            recall_count += 1
                            logger.debug(f"成功撤回消息 {message_id}")
                        await asyncio.sleep(0.1)  # 避免频繁调用API
                    except Exception as e:
                        logger.debug(f"撤回消息 {message_id} 失败: {e}")
                        continue
            
            logger.info(f"步骤4完成: 已撤回 {recall_count} 条消息")
            
            # 5. 清理过期消息
            current_time = time.time()
            logger.info(f"步骤5: 清理群聊 {group_id} 的过期消息")
            self._cleanup_expired_messages(group_id, current_time)
            
            # 6. 发送警告消息
            alert_message = self._get_config_value("SPAM_ALERT_MESSAGE",
                "⚠️ 检测到疑似推销信息，相关消息已被处理，用户已被禁言。")
            logger.info(f"步骤6: 发送警告消息")
            
            # 返回警告消息结果
            return event.plain_result(alert_message)
            
        except Exception as e:
            logger.error(f"处理推销消息时出错: {e}", exc_info=True)
            return event.plain_result("❌ 处理推销消息时发生错误，请检查日志")
    
    async def _try_recall_message_by_id(self, event: AstrMessageEvent, message_id: str) -> bool:
        """尝试根据消息ID撤回消息"""
        try:
            platform_name = event.get_platform_name()
            if platform_name == "aiocqhttp":
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                if isinstance(event, AiocqhttpMessageEvent):
                    client = event.bot
                    payloads = {
                        "message_id": message_id,
                    }
                    ret = await client.api.call_action('delete_msg', **payloads)
                    logger.debug(f"撤回消息 {message_id} 返回: {ret}")
                    return True
            return False
        except Exception as e:
            logger.debug(f"撤回消息 {message_id} 失败: {e}")
            return False
    
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
                            "group_id": str(group_id),
                            "user_id": str(user_id),
                            "duration": duration  # 禁言时长（秒）
                        }
                        logger.debug(f"调用 set_group_ban API，payloads: {payloads}")
                        ret = await client.api.call_action('set_group_ban', **payloads)
                        logger.debug(f"禁言用户 {user_id} 返回: {ret}")
                        
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
    
    async def _get_group_name(self, group_id: str) -> str:
        """获取群聊名称"""
        try:
            # 尝试从事件信息中获取群聊名称
            platform_meta = self.context.cached_platform_meta
            if platform_meta and hasattr(platform_meta, 'aiocqhttp'):
                adapter = platform_meta.aiocqhttp
                if adapter:
                    try:
                        # 调用 get_group_info API 获取群信息
                        group_info = await adapter.call_api("get_group_info", group_id=str(group_id))
                        if group_info and 'group_name' in group_info:
                            group_name = group_info['group_name']
                            logger.debug(f"获取到群聊名称: {group_name} (群聊ID: {group_id})")
                            return group_name
                    except Exception as e:
                        logger.debug(f"获取群聊名称失败: {e}")
            
            # 如果无法获取群聊名称，返回默认值
            return "未知群聊"
            
        except Exception as e:
            logger.warning(f"获取群聊名称时出错: {e}")
            return "未知群聊"
    
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
            
            # 获取消息ID
            raw_msg = getattr(event.message_obj, 'raw_message', {})
            msg_id = None
            if isinstance(raw_msg, dict) and 'message_id' in raw_msg:
                msg_id = raw_msg['message_id']
            else:
                msg_id = getattr(event.message_obj, 'message_id', '')
            
            # 将消息添加到对应群聊的消息池
            self._add_message_to_pool(group_id, user_id, message_content, timestamp, str(msg_id) if msg_id else "")
            logger.debug(f"已将消息添加到群聊 {group_id} 用户 {user_id} 的消息池")
                
            # 检查队列大小，避免积压过多
            max_queue_size = int(self._get_config_value("MAX_DETECTION_QUEUE_SIZE", 50))
            if self.detection_queue.qsize() >= max_queue_size:
                logger.warning(f"检测队列已满 ({self.detection_queue.qsize()})，跳过当前消息")
                return
            
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
                    logger.info(f"图片内容提取成功: {image_content[:100]}...")
                else:
                    logger.debug("图片内容提取失败或无内容")
            
            # 将检测任务加入队列：(群聊ID, 用户ID, 用户名, 消息内容, 发送时间, 事件对象, 图片内容)
            logger.debug(f"将消息加入检测队列: {message_content[:50]}...")
            detection_task = (group_id, user_id, user_name, message_content, timestamp, event, image_content)
            await self.detection_queue.put(detection_task)
            logger.debug(f"消息已加入队列，当前队列大小: {self.detection_queue.qsize()}")
                
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
            # 使用批量检测方法测试单条消息
            test_user_id = "test_user"
            test_batch_input = {test_user_id: message}
            spam_user_ids = await self._batch_spam_detection(test_batch_input)
            is_spam = test_user_id in spam_user_ids
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
            
            group_whitelist = self._get_config_value("WHITELIST_GROUPS", [])
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
            
            # 检查消息池状态
            total_groups = len(self.group_message_pools)
            total_users = sum(len(users) for users in self.group_message_pools.values())
            total_messages = sum(
                len(messages) for group in self.group_message_pools.values() 
                for messages in group.values()
            )
            config_status.append(f"消息池: {total_groups} 个群聊, {total_users} 个用户, {total_messages} 条消息")
            config_status.append(f"检测队列: {self.detection_queue.qsize()} 个待处理任务")
            
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
        logger.info("防推销插件正在停止...")
        
        # 停止队列处理器
        self.detection_worker_running = False
        
        # 等待队列中剩余任务完成
        if not self.detection_queue.empty():
            logger.info(f"等待队列中剩余 {self.detection_queue.qsize()} 个任务完成...")
            await self.detection_queue.join()
        
        # 清理消息池
        self.group_message_pools.clear()
        
        logger.info("防推销插件已停止")
