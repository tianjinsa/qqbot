import asyncio
import time
import base64
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from networkx import nodes
from openai import AsyncOpenAI

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig
import astrbot.api.message_components as Comp


@register("astrbot_plugin_spam_detector", "AstrBot Dev Team", "智能防推销插件，使用AI检测并处理推销信息", "1.2.3", "https://github.com/tianjinsa/qqbot/tree/main/astrbot_plugin_spam_detector")
class SpamDetectorPlugin(Star):
    def jsonout(self, data_to_serialize, log_prefix="JSON Output"): # 增加log_prefix参数，方便区分日志来源
        """
        将给定的数据序列化为 JSON 格式并记录到日志中。
        data_to_serialize: 可能是列表、Pydantic 模型或其他可序列化对象。
        """
        try:
            processed_data = None
            if isinstance(data_to_serialize, list):
                # 如果是列表，遍历列表中的每个元素并尝试转换
                serializable_list_items = []
                for item in data_to_serialize:
                    if isinstance(item, BaseModel): # 如果是 Pydantic 模型
                        serializable_list_items.append(item.dict())
                    elif hasattr(item, 'dict') and callable(item.dict): # 可能是类似Pydantic但不是其子类的对象
                        serializable_list_items.append(item.dict())
                    elif hasattr(item, '__dict__'): # 尝试序列化对象的__dict__
                        try:
                            # 注意：__dict__可能包含非JSON序列化对象，可能会再次引发TypeError
                            # Pydantic模型通常不直接用__dict__，而是用.dict()
                            serializable_list_items.append(item.__dict__)
                        except Exception as e:
                            logger.warning(f"jsonout: Could not convert item.__dict__ for JSON serialization: {e}. Using str(item).")
                            serializable_list_items.append(str(item))
                    else:
                        # 假设 item 已经是字典或基本类型，或者我们接受它的 str 表示
                        # 如果 Node 对象落到这里，会导致错误，所以需要确保 Node 被上面分支处理
                        serializable_list_items.append(item)
                processed_data = serializable_list_items
            elif isinstance(data_to_serialize, BaseModel): # 如果是单个 Pydantic 模型
                processed_data = data_to_serialize.dict()
            elif hasattr(data_to_serialize, 'dict') and callable(data_to_serialize.dict):
                processed_data = data_to_serialize.dict()
            elif hasattr(data_to_serialize, '__dict__'):
                try:
                    processed_data = data_to_serialize.__dict__
                except Exception as e:
                    logger.warning(f"jsonout: Could not convert data_to_serialize.__dict__ for JSON serialization: {e}. Using str(data_to_serialize).")
                    processed_data = str(data_to_serialize)
            else:
                # 假设 data_to_serialize 已经是字典或基本类型
                processed_data = data_to_serialize
            json_output = json.dumps(processed_data, indent=2, ensure_ascii=False)
            logger.info(f"{log_prefix}: \n{json_output}")
        except TypeError as e:
            logger.error(f"{log_prefix}: Failed to serialize data to JSON due to TypeError: {e}")
            logger.error(f"{log_prefix}: Original Data (repr): {repr(data_to_serialize)}")
        except Exception as e:
            logger.error(f"{log_prefix}: An unexpected error occurred during JSON serialization: {e}")
            logger.error(f"{log_prefix}: Original Data (repr): {repr(data_to_serialize)}")
        return
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
        # 用户处理锁：防止同一用户被并发处理
        self.processing_users = set()  # 存储 (group_id, user_id)
        # AI调用并发控制
        self.ai_semaphore = None  # 在initialize中初始化
        # 消息池锁，防止并发修改
        self.message_pool_lock = asyncio.Lock()
        
    async def initialize(self):
        """插件初始化"""
        logger.info("防推销插件已启动")
        # 初始化AI调用并发限制
        max_concurrent_ai_calls = int(self._get_config_value("MAX_CONCURRENT_AI_CALLS", 2))
        self.ai_semaphore = asyncio.Semaphore(max_concurrent_ai_calls)
        logger.info(f"AI调用并发限制已设置为: {max_concurrent_ai_calls}")
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
                batch_wait_time = float(self._get_config_value("BATCH_WAIT_TIME", 5.0))
                
                # 等待队列中的检测任务或超时触发批量处理
                try:
                    detection_task = await asyncio.wait_for(self.detection_queue.get(), timeout=batch_wait_time)
                    timeout_occurred = False
                except asyncio.TimeoutError:
                    timeout_occurred = True
                
                if not timeout_occurred:
                    # 正常获取到任务
                    group_id, user_id, user_name, message_content, timestamp, image_urls, event = detection_task
                    # 初始化群聊的批量缓冲区
                    if group_id not in self.batch_buffer:
                        self.batch_buffer[group_id] = []
                        self.batch_timer[group_id] = time.time()
                    # 添加任务到批量缓冲区
                    self.batch_buffer[group_id].append(detection_task)
                    # 标记任务完成
                    self.detection_queue.task_done()

                # 超时或新任务到达后，检查所有群聊的批量缓冲区
                now = time.time()
                # 获取最大字符长度配置
                max_chars = int(self._get_config_value("BATCH_MAX_TEXT_LENGTH", 5000))
                for gid, tasks in list(self.batch_buffer.items()):
                    if not tasks:
                        continue
                    # 计算文本总字符数（图片内容视为0字符）
                    total_chars = sum(len(task[3] or "") for task in tasks)
                    # 达到批量大小、超时或字符数超限时触发处理
                    if len(tasks) >= batch_size or now - self.batch_timer[gid] > batch_wait_time or total_chars > max_chars:
                        # 速率限制：确保距离上次调用至少 rate_limit 秒
                        time_since_last_call = now - self.last_model_call_time
                        if time_since_last_call < rate_limit:
                            await asyncio.sleep(rate_limit - time_since_last_call)
                        # 更新最后调用时间
                        self.last_model_call_time = time.time()
                        # 准备处理批量任务
                        tasks_to_process = tasks.copy()
                        self.batch_buffer[gid].clear()
                        self.batch_timer[gid] = time.time()
                        await self._process_batch_tasks(gid, tasks_to_process)
                
            except asyncio.CancelledError:
                logger.info("推销检测队列处理器被取消")
                break
            except Exception as e:
                logger.error(f"队列处理器出错: {e}", exc_info=True)
                await asyncio.sleep(1)  # 出错时暂停1秒
        
        logger.info("推销检测队列处理器已停止")
    
    async def _process_batch_tasks(self, group_id: str, tasks: List[tuple]):
        """批量处理同一群聊的检测任务"""
        # 如果没有任务则直接返回
        if not tasks:
            return
        logger.info(f"开始批量处理群聊 {group_id} 的 {len(tasks)} 条消息")
        try:
            # 直接将所有任务作为一个批量处理
            await self._process_task_batch(tasks, group_id, "预批量")
        except Exception as e:
            logger.error(f"批量处理任务时出错: {e}", exc_info=True)
    
    async def _build_full_content(self, task: tuple) -> str:
        """构建完整的消息内容（文本+图片），使用预提取的图片URL列表"""
        _, user_id, user_name, message_content, timestamp, image_urls, event = task
        full_content = message_content
        
        # 如果有图片URL列表，提取图片内容
        if image_urls:
            try:
                logger.debug(f"提取 {len(image_urls)} 张图片的内容")
                image_content = await self._extract_image_content(image_urls)
                if image_content:
                    full_content = f"{message_content}\n[图片内容]: {image_content}"
            except Exception as e:
                logger.warning(f"提取图片内容失败: {e}")
        
        return full_content
    
    def _extract_task_info(self, task: tuple) -> tuple:
        """提取任务信息"""
        _, user_id, user_name, message_content, timestamp, image_urls, event = task
        return user_id, user_name, message_content, timestamp, image_urls, event
    
    async def _process_task_batch(self, tasks: List[tuple], group_id: str, batch_type: str):
        """处理一批任务"""
        if not tasks:
            return
            
        logger.info(f"开始{batch_type}处理 {len(tasks)} 条消息")
        
        # 构建批量输入（同一用户的多条消息合并）
        batch_input: Dict[str, str] = {}
        task_map: Dict[str, tuple] = {}
        users_to_lock = set()  # 需要加锁的用户
        num=0
        for task in tasks:
            user_id, user_name, message_content, timestamp, image_urls, event = self._extract_task_info(task)
            user_lock = (group_id, user_id)
            # 跳过已在处理中的用户
            if user_lock in self.processing_users:
                logger.info(f"用户 {user_name} ({user_id}) 已在处理中，跳过检测")
                continue
            # 获取完整内容
            full_content = await self._build_full_content(task)
            # 合并同一用户的消息内容
            if user_id in batch_input:
                batch_input[user_id] += f"\n{full_content}"
            else:
                batch_input[user_id] = full_content
                # 记录首次出现的任务用于后续处理
                task_map[user_id] = task
            users_to_lock.add(user_lock)
            num+=1
        
        if not batch_input:
            logger.info(f"{batch_type}处理完成：所有任务都被跳过或无效")
            return
        
        # 为所有用户加锁
        for user_lock in users_to_lock:
            self.processing_users.add(user_lock)
        
        try:
            # 批量检测
            total_chars = sum(len(content) for content in batch_input.values())
            logger.info(f"{batch_type}检测 {num} 条消息，总字符数: {total_chars}")
            
            spam_user_ids = await self._batch_spam_detection(batch_input)
            
            # 处理检测结果
            for user_id in spam_user_ids:
                if user_id in task_map:
                    task = task_map[user_id]
                    user_id, user_name, message_content, timestamp, image_urls, event = self._extract_task_info(task)
                    await self._handle_spam_detection_result(user_id, user_name, group_id, event, batch_type)
            
            logger.info(f"{batch_type}处理完成，发现 {len(spam_user_ids)} 个推销用户，分别是: {', '.join(spam_user_ids)}")
        finally:
            # 确保处理完成后移除所有锁
            for user_lock in users_to_lock:
                self.processing_users.discard(user_lock)
    
    async def _handle_spam_detection_result(self, user_id: str, user_name: str, group_id: str, event, context: str):
        """处理推销检测结果"""
        logger.info(f"🚨 {context}检测到推销消息，用户: {user_name} ({user_id}), 群聊: {group_id}")
        result = await self._handle_spam_message_new(event, group_id, user_id, user_name)
        if result:
            await event.send(result)
    
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

请分析所有消息，找出其中的推销信息，并返回一个JSON格式的结果(不是md格式)，格式为：{"y":[用户ID1,用户ID2,...]}
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
                    # 清理可能的markdown代码块格式
                    cleaned_result = result.strip()
                    if cleaned_result.startswith("```json"):
                        cleaned_result = cleaned_result[7:]  # 移除开头的```json
                    if cleaned_result.endswith("```"):
                        cleaned_result = cleaned_result[:-3]  # 移除结尾的```
                    cleaned_result = cleaned_result.strip()
                    
                    # 解析JSON结果
                    result_json = json.loads(cleaned_result)
                    spam_user_ids = result_json.get("y", [])
                    
                    # 确保返回的是字符串列表
                    spam_user_ids = [str(uid) for uid in spam_user_ids]
                    
                    # logger.info(f"批量推销检测模型返回结果: {spam_user_ids}")
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
        async with self.ai_semaphore:  # 并发控制
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
                    raw_content = response.choices[0].message.content
                    cleaned_content = self._remove_thinking_tags(raw_content)
                    logger.debug(f"文本模型调用成功，返回内容: {cleaned_content[:100]}...")
                    return cleaned_content
                else:
                    logger.warning("文本模型返回空内容")
                    
            except Exception as e:
                logger.error(f"文本模型调用失败: {e}", exc_info=True)
            
            return None
    
    async def _call_vision_model(self, messages: List[Dict], model_id: str = None) -> Optional[str]:
        """调用视觉模型"""
        async with self.ai_semaphore:  # 并发控制
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
                    logger.debug("使用extra_body方式关闭thinking模式")
                    response = await client.chat.completions.create(
                        **api_params,
                        extra_body={"thinking": {"type": "disabled"}}
                    )
                    logger.debug("成功使用extra_body方式关闭thinking模式")

                if response.choices and len(response.choices) > 0:
                    raw_content = response.choices[0].message.content
                    cleaned_content = self._remove_thinking_tags(raw_content)
                    logger.debug(f"视觉模型调用成功，返回内容: {cleaned_content[:100]}...")
                    return cleaned_content
                else:
                    logger.warning("视觉模型返回空内容")
                    
            except Exception as e:
                logger.error(f"视觉模型调用失败: {e}", exc_info=True)
            
            return None
    
    def _remove_thinking_tags(self, content: str) -> str:
        """
        移除模型响应中的思考标签
        
        Args:
            content: 原始模型响应内容
            
        Returns:
            str: 去除思考标签后的内容
        """
        if not isinstance(content, str):
            return content
            
        import re
        # 使用正则表达式匹配并移除 <think>...</think> 标签及其内容
        # 使用 re.DOTALL 标志使 . 能匹配换行符
        cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理多余的空白字符
        cleaned_content = re.sub(r'\n\s*\n', '\n', cleaned_content.strip())
        
        return cleaned_content
        
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
    
    def _is_group_blacklisted(self, group_id: str) -> bool:
        """检查群聊是否在黑名单中"""
        if not group_id:
            return False
        
        blacklist = self._get_config_value("BLACKLIST_GROUPS", [])
        if isinstance(blacklist, str):
            # 如果是字符串，按逗号分割
            blacklist = [gid.strip() for gid in blacklist.split(",") if gid.strip()]
        
        # 如果白名单为空，则检测所有群聊
        if not blacklist:
            return True
        
        return group_id in blacklist
    
    def _add_message_to_pool(self, group_id: str, user_id: str, timestamp: float, 
                            message_id: str = "", original_messages = None):
        """将消息添加到对应群聊的消息池中（不存储消息内容，只存储原始组件）"""
        # 确保参数类型正确
        group_id = str(group_id)
        user_id = str(user_id)
        
        if group_id not in self.group_message_pools:
            self.group_message_pools[group_id] = {}
        
        if user_id not in self.group_message_pools[group_id]:
            self.group_message_pools[group_id][user_id] = []
        
        # 添加消息记录（不存储content，因为有原始组件提供）
        message_record = {
            "timestamp": timestamp,
            "message_id": str(message_id) if message_id else "",
            "recalled": False,
            "original_messages": original_messages or []  # 存储原始消息组件用于转发
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
        # 确保参数类型正确
        group_id = str(group_id)
        user_id = str(user_id)
        
        if group_id not in self.group_message_pools:
            logger.debug(f"群聊 {group_id} 不在消息池中")
            return []
        
        if user_id not in self.group_message_pools[group_id]:
            logger.debug(f"用户 {user_id} 在群聊 {group_id} 中没有消息")
            return []
        
        user_messages = self.group_message_pools[group_id][user_id].copy()
        logger.debug(f"从群聊 {group_id} 用户 {user_id} 获取到 {len(user_messages)} 条消息")
        return user_messages
    
    def _pop_user_messages_from_pool(self, group_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        原子地获取并移除指定用户在群聊中的所有消息记录。
        这可以防止在处理期间，消息池被其他并发任务修改，从而保证操作的原子性。
        返回一个包含用户消息记录的【数据快照】。
        """
        # 确保参数类型正确
        group_id = str(group_id)
        user_id = str(user_id)
        
        # 检查群聊和用户是否存在于消息池中
        if group_id in self.group_message_pools and user_id in self.group_message_pools[group_id]:
            # 使用 pop 方法。这是一个原子操作：如果键存在，它会移除该键并返回其值。
            # 这就确保了一旦一个处理流程拿到了数据，其他流程就拿不到了。
            user_messages = self.group_message_pools[group_id].pop(user_id, [])
            # logger.info(f"已从消息池中取出并隔离了用户 {user_id} 的 {len(user_messages)} 条消息进行处理。")
            
            # 清理空群聊：如果 pop 操作后该群聊没有任何用户记录了，就从池中删除该群聊
            if not self.group_message_pools[group_id]:
                self.group_message_pools.pop(group_id)
                logger.debug(f"群聊 {group_id} 已从消息池中清理（无用户消息）")
                
            return user_messages
        
        # 如果用户或群聊一开始就不在池中，返回空列表
        logger.debug(f"用户 {user_id} 在群聊 {group_id} 中没有消息记录")
        return []
    
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
            # 将队列中需要保留的任务暂存到列表中
            tasks_to_keep = []
            cleared_count = 0
            
            while not self.detection_queue.empty():
                try:
                    task = self.detection_queue.get_nowait()
                    task_group_id, task_user_id = task[0], task[1]
                    
                    if task_group_id == group_id and task_user_id == user_id:
                        cleared_count += 1
                        logger.debug(f"从队列中清除任务: 群聊{task_group_id}, 用户{task_user_id}")
                    else:
                        tasks_to_keep.append(task)
                except asyncio.QueueEmpty:
                    # 在并发环境下，队列可能在检查后变空
                    break
            
            # 将保留的任务放回队列
            for task in tasks_to_keep:
                self.detection_queue.put_nowait(task)
            
            # if cleared_count > 0:
                # logger.info(f"已从检测队列中清除 {cleared_count} 个重复任务 (群聊: {group_id}, 用户: {user_id})")
                
        except Exception as e:
            logger.error(f"清理检测队列时出错: {e}", exc_info=True)
    
    async def _forward_messages_as_merged(self, admin_chat_id: str, group_id: str, user_id: str, 
                                        user_name: str, user_messages: List[Dict], event: AstrMessageEvent):
        """使用合并转发的方式将消息转发到管理员群"""
        try:
            if not admin_chat_id:
                logger.warning("管理员群聊ID未配置，无法转发消息")
                return
                
            # 确保参数类型正确
            group_id = str(group_id)
            user_id = str(user_id)
            admin_chat_id = str(admin_chat_id)
            
            if not user_messages:
                logger.warning(f"没有找到属于用户 {user_id} 的消息，跳过转发")
                return

            logger.info(f"准备转发 {len(user_messages)} 条属于用户 {user_id} 的消息到管理员群")

            # 检查事件类型
            if not hasattr(event, 'bot'):
                logger.warning("事件对象没有bot属性，无法使用合并转发")
                await self._forward_to_admin_text(admin_chat_id, group_id, user_id, user_name, user_messages, event)
                return
            
            client = event.bot
            group_name = await self._get_group_name(event, group_id)
            
            # 每次都重新构建合并转发的节点列表，确保不影响后续转发
            nodes = []  # 每次都创建新的节点列表
            
            # 添加标题节点
            title_content = f"🚨 推销检测报告\n👤 用户: {user_name} ({user_id})\n🏷️ 原群聊: {group_name} ({group_id})\n⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # 获取机器人ID，处理可能的functools.partial对象
            bot_id = getattr(client, 'self_id', '0')
            if callable(bot_id):
                try:
                    bot_id = str(bot_id())
                except:
                    bot_id = '0'
            else:
                bot_id = str(bot_id)
            
            # 创建新的标题节点
            title_node = Comp.Node(
                uin=bot_id,
                name="AstrBot反推销系统",
                content=[Comp.Plain(title_content)]
            )
            nodes.append(title_node)
            
            # 添加每条被撤回的消息作为节点
            for i, msg_record in enumerate(user_messages):
                timestamp_str = datetime.fromtimestamp(msg_record.get("timestamp", time.time())).strftime('%H:%M:%S')
                
                # 获取原始消息组件
                original_messages = msg_record.get("original_messages", [])
                if original_messages:
                    # 先添加一个AstrBot系统发送的时间戳节点
                    timestamp_node = Comp.Node(
                        uin=bot_id,
                        name="AstrBot反推销系统",
                        content=[Comp.Plain(f"消息时间: {timestamp_str}")]
                    )
                    nodes.append(timestamp_node)
                    
                    comp_node = Comp.Node(
                        uin=str(user_id),
                        name=f"{user_name}",
                        content=original_messages  # 每个组件作为单独的节点内容
                    )
                    nodes.append(comp_node)
                else:
                    # 如果没有原始组件，不添加内容，并报错，说明有重复检测
                    logger.warning(f"用户 {user_id} 的消息记录中缺少原始组件，可能是重复检测")
            
            if len(nodes) <= 1:
                logger.warning("没有有效的消息内容，跳过合并转发")
                return
            
            # 发送合并转发
            # logger.info(f"发送合并转发到管理员群 {admin_chat_id}，包含 {len(nodes)} 个节点")
            
            # 直接发送合并转发，让底层自动处理所有消息格式
            platform_name = event.get_platform_name()
            if platform_name == "aiocqhttp":
                client = event.bot
                
                # 构建原生转发消息，直接使用Node的content
                forward_msg = []
                for node in nodes:
                    forward_msg.append({
                        "type": "node",
                        "data": {
                            "uin": str(node.uin),
                            "name": node.name,
                            "content": node.content  # 直接使用Node的content，让CQHTTP自动处理
                        }
                    })

                ret = await client.api.call_action(
                    'send_group_forward_msg',
                    group_id=str(admin_chat_id),
                    messages=forward_msg
                )
                # logger.info(f"合并转发结果: {ret}")
            else:
                logger.warning(f"平台 {platform_name} 不支持合并转发")
                await self._forward_to_admin_text(admin_chat_id, group_id, user_id, user_name, user_messages, event)
                return
            
            # 显式清理节点列表，确保不影响后续转发
            for node in nodes:
                node.content.clear() if hasattr(node.content, 'clear') else None
            nodes.clear()
            logger.debug("合并转发节点列表已清理")
            
        except Exception as e:
            logger.error(f"合并转发失败: {e}", exc_info=True)
            # 回退到文本转发
            await self._forward_to_admin_text(admin_chat_id, group_id, user_id, user_name, user_messages, event)
        finally:
            # 最终清理，确保节点列表不会保留
            try:
                if 'nodes' in locals():
                    for node in nodes:
                        if hasattr(node, 'content') and hasattr(node.content, 'clear'):
                            node.content.clear()
                    nodes.clear()
                    logger.debug("finally块中清理了合并转发节点列表")
            except:
                pass
    
    async def _forward_to_admin_text(self, admin_chat_id: str, group_id: str, user_id: str,
                                   user_name: str, user_messages: List[Dict], event: AstrMessageEvent):
        """文本形式转发到管理员群（作为合并转发的备用方案）"""
        try:
            # 确保参数类型正确
            group_id = str(group_id)
            user_id = str(user_id)
            admin_chat_id = str(admin_chat_id)
            
            if not user_messages:
                logger.warning(f"没有找到属于用户 {user_id} 的消息，跳过文本转发")
                return
            
            group_name = await self._get_group_name(event, group_id)
            
            # 构建转发内容
            forward_content = f"🚨 推销检测报告\n"
            forward_content += f"👤 用户: {user_name} ({user_id})\n"
            forward_content += f"🏷️ 原群聊: {group_name} ({group_id})\n"
            forward_content += f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            forward_content += f"📋 被撤回的消息 ({len(user_messages)} 条):\n"
            
            for i, msg_record in enumerate(user_messages, 1):
                timestamp_str = datetime.fromtimestamp(msg_record.get("timestamp", time.time())).strftime('%H:%M:%S')
                # 从原始组件构建简单的文本表示
                original_messages = msg_record.get("original_messages", [])
                if original_messages:
                    content_text = self._build_simple_text_from_components(original_messages)
                    if content_text.strip():
                        forward_content += f"{i}. [{timestamp_str}] {content_text}\n"
                    else:
                        forward_content += f"{i}. [{timestamp_str}] [复杂消息内容]\n"
                else:
                    forward_content += f"{i}. [{timestamp_str}] [消息内容已清理]\n"
            
            platform_name = event.get_platform_name()
            if platform_name == "aiocqhttp":
                client = event.bot
                ret = await client.api.call_action(
                    'send_group_msg',
                    group_id=admin_chat_id,
                    message=forward_content
                )
                # logger.info(f"文本转发结果: {ret}")
            
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
            # logger.info(f"开始处理 {len(image_urls)} 张图片以提取内容")
            # 处理图片URL，支持HTTP链接和本地文件路径转base64
            processed_images = []
            for i, url in enumerate(image_urls):
                logger.debug(f"处理图片 {i+1}/{len(image_urls)}: {url}")
                
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
            
            # 获取视觉模型系统提示词配置
            system_prompt = self._get_config_value("VISION_MODEL_SYSTEM_PROMPT", "你是一个图片内容识别助手，请客观描述图片内容，特别是提取其中的文字信息。")
            user_prompt = self._get_config_value("VISION_MODEL_USER_PROMPT", "请描述这张图片的主要内容，特别是如果有文字请完整提取出来。")
            
            # 构建符合视觉模型格式的消息
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
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
            # logger.info(f"视觉模型返回结果: {result}..." if result else "视觉模型未返回结果")
            return result or ""
            
        except Exception as e:
            logger.error(f"图片内容提取失败: {e}")
            return ""
    
    async def _handle_spam_message_new(self, event: AstrMessageEvent, group_id: str, user_id: str, user_name: str) -> Optional[Comp.BaseMessageComponent]:
        """处理检测到的推销消息 - 新的、并发安全的逻辑流程"""
        try:
            logger.info(f"开始处理推销消息，用户: {user_name} ({user_id})，群聊: {group_id}")
            
            # 步骤 0: 清理检测队列中的重复任务 (此逻辑保留)
            self._clear_user_detection_queue(group_id, user_id)
            
            # 步骤 1: 【核心修改】原子地从消息池获取并移除该用户的所有消息，形成数据快照
            # 这一步是实现并发安全的关键。
            user_messages_snapshot = self._pop_user_messages_from_pool(group_id, user_id)
            
            # 如果快照为空，说明消息已被其他并发任务处理，或已过期被清理。立即终止。
            if not user_messages_snapshot:
                logger.warning(f"处理用户 {user_id} 时，其消息已不在池中。可能已被其他任务处理或已过期。终止当前处理流程。")
                return None # 必须返回，防止重复操作

            # logger.info(f"步骤1: 已隔离用户 {user_id} 的 {len(user_messages_snapshot)} 条消息作为处理快照。")

            # 步骤 2: 禁言用户
            mute_duration = self._get_config_value("MUTE_DURATION", 600)
            # logger.info(f"步骤2: 禁言用户 {user_id}，时长: {mute_duration} 秒")
            await self._try_mute_user(event, user_id, mute_duration)
            
            # 步骤 3: 进行合并转发。现在基于隔离的、绝对安全的【数据快照】进行。
            admin_chat_id = self._get_config_value("ADMIN_CHAT_ID", "")
            if admin_chat_id:
                # logger.info(f"步骤3: 合并转发推销消息到管理员群: {admin_chat_id}")
                # 传入的是我们刚刚隔离的、安全的 user_messages_snapshot 局部变量
                await self._forward_messages_as_merged(admin_chat_id, group_id, user_id, user_name, user_messages_snapshot, event)
            else:
                logger.warning("步骤3: 管理员群聊ID未配置，跳过转发")

            # 步骤 4: 执行消息撤回，同样基于安全的【数据快照】
            # logger.info(f"步骤4: 开始撤回用户 {user_id} 的消息")
            recall_count = 0
            for message_record in user_messages_snapshot: # 遍历的是安全的快照
                message_id = message_record.get("message_id")
                if message_id:
                    try:
                        success = await self._try_recall_message_by_id(event, message_id)
                        if success:
                            recall_count += 1
                        await asyncio.sleep(0.1)  # 保留API调用间隔
                    except Exception as e:
                        logger.debug(f"撤回消息 {message_id} 失败: {e}")
                        continue
            
            logger.info(f"步骤4完成: 共撤回用户 {user_id} 的 {recall_count} 条消息")
            
            # 步骤 5: 原有的清理逻辑可以移除，因为 _pop_user_messages_from_pool 已经隐式地完成了清理。
            
            # 步骤 6: 发送最终的群内警告消息
            alert_message = self._get_config_value("SPAM_ALERT_MESSAGE",
                "⚠️ 检测到疑似推销信息，相关消息已被处理，用户已被禁言。")
            # logger.info(f"步骤6: 发送警告消息")
            
            return event.plain_result(alert_message)
            
        except Exception as e:
            logger.error(f"处理推销消息时出错: {e}", exc_info=True)
            return event.plain_result("❌ 处理推销消息时发生错误，请检查日志")
    
    async def _try_recall_message_by_id(self, event: AstrMessageEvent, message_id: str) -> bool:
        """尝试根据消息ID撤回消息"""
        try:
            platform_name = event.get_platform_name()
            if platform_name == "aiocqhttp" and hasattr(event, 'bot'):
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
            # logger.info(f"尝试禁言用户 {user_id}，时长: {duration}秒，平台: {platform_name}")
            
            if platform_name == "aiocqhttp" and hasattr(event, 'bot'):
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
    
    def _build_simple_text_from_components(self, original_messages) -> str:
        """从原始消息组件构建简单的文本表示"""
        try:
            text_parts = []
            
            for msg_comp in original_messages:
                # 处理文本消息
                if isinstance(msg_comp, Comp.Plain):
                    text_parts.append(msg_comp.text)
                
                # 处理图片消息
                elif isinstance(msg_comp, Comp.Image):
                    text_parts.append("[图片]")
                
                # 处理合并转发消息
                elif hasattr(msg_comp, 'type') and getattr(msg_comp, 'type', '') == 'forward':
                    text_parts.append("[合并消息无法显示]")
                
                # 检查其他可能的合并转发标识
                elif type(msg_comp).__name__.lower() in ['forward', 'forwardmessage', 'merge', 'mergeforward']:
                    text_parts.append("[合并消息无法显示]")
                
                # 检查是否有forward相关属性
                elif hasattr(msg_comp, 'messages') or (hasattr(msg_comp, 'content') and 
                    isinstance(getattr(msg_comp, 'content'), list) and 
                    len(getattr(msg_comp, 'content')) > 0):
                    text_parts.append("[合并消息无法显示]")
                
                # 处理其他类型的消息组件
                else:
                    # 尝试获取文本表示
                    if hasattr(msg_comp, 'text'):
                        text_parts.append(msg_comp.text)
                    elif hasattr(msg_comp, '__str__'):
                        comp_str = str(msg_comp)
                        if not comp_str.startswith('<'):
                            text_parts.append(comp_str)
                        else:
                            text_parts.append(f"[{type(msg_comp).__name__}]")
                    else:
                        text_parts.append(f"[{type(msg_comp).__name__}]")
            
            return ' '.join(text_parts)
            
        except Exception as e:
            logger.warning(f"构建简单文本时出错: {e}")
            return "[消息内容解析失败]"
    
    def _extract_content_from_messages(self, original_messages: List[Any]) -> tuple[str, List[str]]:
        """
        从original_messages中按深度优先提取文本和图片URL，使用多次遍历减少内存占用
        
        Args:
            original_messages: 处理后的消息组件列表
            
        Returns:
            tuple[str, List[str]]: (合并的文本内容, 图片URL列表)
        """
        max_text_length = int(self._get_config_value("BATCH_MAX_TEXT_LENGTH", 5000))
        max_images = int(self._get_config_value("FORWARD_MESSAGE_MAX_IMAGES", 10))
        max_depth = int(self._get_config_value("FORWARD_MESSAGE_MAX_DEPTH", 3))
        max_depth+=1
        text_parts = []
        image_urls = []
        current_text_length = 0
        
        def extract_content_at_depth(components: List[Any], target_depth: int, current_depth: int = 0) -> tuple[List[str], List[str]]:
            """在指定深度提取内容"""
            texts = []
            images = []
            
            for comp in components:
                if current_depth == target_depth:
                    # 当前深度匹配，提取内容
                    if isinstance(comp, Comp.Plain):
                        text = comp.text.strip()
                        if text:
                            texts.append(text)
                    elif isinstance(comp, Comp.Image):
                        url = getattr(comp, 'url', None) or getattr(comp, 'file', None)
                        if url:
                            images.append(url)
                else:
                    # 需要继续深入
                    if isinstance(comp, Comp.Node):
                        node_content = getattr(comp, 'content', [])
                        if node_content:
                            sub_texts, sub_images = extract_content_at_depth(
                                node_content, target_depth, current_depth + 1
                            )
                            texts.extend(sub_texts)
                            images.extend(sub_images)
                    elif hasattr(comp, 'content') and isinstance(comp.content, list):
                        sub_texts, sub_images = extract_content_at_depth(
                            comp.content, target_depth, current_depth + 1
                        )
                        texts.extend(sub_texts)
                        images.extend(sub_images)
            
            return texts, images
        
        # 按深度逐层提取内容，深度越小优先级越高
        for depth in range(max_depth):
            # logger.info(f"正在提取第{depth}层内容")
            
            # 提取当前深度的所有文本和图片
            depth_texts, depth_images = extract_content_at_depth(original_messages, depth)
            
            # 添加文本内容（检查长度限制）
            for text in depth_texts:
                if current_text_length + len(text) <= max_text_length:
                    text_parts.append(text)
                    current_text_length += len(text)
                else:
                    # 即使超出限制，也添加当前文本，然后停止
                    text_parts.append(text)
                    logger.info(f"在第{depth}层达到文本长度限制，停止提取")
                    break
            
            # 添加图片URL（检查数量限制）
            for img_url in depth_images:
                if len(image_urls) < max_images:
                    image_urls.append(img_url)
                else:
                    break
            
            # 检查是否已达到所有限制
            if current_text_length >= max_text_length and len(image_urls) >= max_images:
                logger.debug(f"在第{depth}层达到所有限制，停止提取")
                break
            
            # 如果当前深度没有内容，也可以继续下一深度
            if not depth_texts and not depth_images:
                logger.debug(f"第{depth}层无内容，继续下一层")
                continue
        
        combined_text = '\n'.join(text_parts)
        logger.info(f"从消息中提取文本长度: {len(combined_text)}, 图片数量: {len(image_urls)}")
        
        return combined_text, image_urls
    
    async def _process_forward_message_recursive(self, nested_data: List[Dict], timestamp: int, depth: int = 0, max_depth: int = 3) -> List[Any]:
        """
        递归处理合并转发消息，支持多层嵌套
        
        Args:
            nested_data: 预先获取的嵌套消息数据结构
            timestamp: 时间戳
            depth: 当前递归深度
            max_depth: 最大递归深度
            
        Returns:
            List[Any]: 处理后的消息组件列表
        """
        if depth >= max_depth:
            logger.warning(f"达到最大递归深度 {max_depth}，停止处理")
            return [Comp.Plain("[嵌套消息过深，无法显示]")]
        
        if not nested_data:
            logger.debug(f"第{depth}层嵌套数据为空，返回")
            return [Comp.Plain("[嵌套消息为空，无法显示]")]
        
        processed_nodes = []
        
        try:
            logger.debug(f"处理第{depth}层嵌套数据，包含 {len(nested_data)} 个节点")
            
            for node in nested_data:
                timesend = node.get('time', timestamp)
                sender = node.get('sender', {})
                uint = str(sender.get('user_id', ''))
                namet = sender.get('nickname', '')
                content_list = node.get('message', [])
                
                # 处理节点内的消息组件
                comps = []
                for seg in content_list:
                    t = seg.get('type')
                    d = seg.get('data', {})
                    
                    if t in ('text', 'plain'):
                        comps.append(Comp.Plain(d.get('text', '')))
                    elif t == 'image':
                        comps.append(Comp.Image(url=d.get('url'), file=d.get('file')))
                    elif t == 'at':
                        comps.append(Comp.At(qq=d.get('qq')))
                    elif t == 'forward':
                        # 发现嵌套的合并消息，从content中获取嵌套结构递归处理
                        nested_content = d.get('content', [])
                        if nested_content:
                            nested_nodes = await self._process_forward_message_recursive(
                                nested_content, timesend, depth + 1, max_depth
                            )
                            comps.extend(nested_nodes)
                        else:
                            logger.warning(f"合并消息缺少嵌套内容，跳过处理")
                            comps.append(seg)
                    else:
                        # 其他类型的消息组件，直接保留
                        comps.append(seg)
                
                # 创建节点组件
                processed_nodes.append(Comp.Node(uin=uint, name=namet, content=comps, time=timesend))
                
        except Exception as e:
            logger.warning(f"处理转发消息失败（深度 {depth}）: {e}")
        
        return processed_nodes
    
    def _should_process_message_type(self, event: AstrMessageEvent) -> bool:
        """检查消息类型是否需要处理（处理文本、图片和合并转发）"""
        try:
            message_components = event.get_messages()
            
            # 检查是否包含可处理的消息类型（文本、图片或合并转发）
            for msg_comp in message_components:
                # 检查是否为文本消息
                if isinstance(msg_comp, Comp.Plain):
                    return True
                
                # 检查是否为图片消息
                elif isinstance(msg_comp, Comp.Image):
                    return True
                
                # 检查是否为合并转发消息
                elif isinstance(msg_comp, Comp.Forward):
                    logger.debug("检测到合并转发消息，将进入处理队列")
                    return True
                elif hasattr(msg_comp, 'type') and getattr(msg_comp, 'type', '') == 'forward':
                    logger.debug("检测到合并转发消息，将进入处理队列")
                    return True
                elif type(msg_comp).__name__.lower() in ['forward', 'forwardmessage', 'merge', 'mergeforward']:
                    logger.debug("检测到合并转发消息，将进入处理队列")
                    return True
            
            # 如果没有找到可识别的组件，检查是否有消息文本
            if event.message_str and event.message_str.strip():
                return True
            
            logger.debug(f"消息类型不需要处理，组件类型: {[type(comp).__name__ for comp in message_components]}")
            return False
            
        except Exception as e:
            logger.warning(f"检查消息类型时出错: {e}")
            # 出错时默认处理
            return True
    
    async def _get_group_name(self, event: AstrMessageEvent, group_id: str) -> str:
        """获取群聊名称"""
        try:
            platform_name = event.get_platform_name()
            if platform_name == "aiocqhttp" and hasattr(event, 'bot'):
                client = event.bot
                group_list = await client.api.call_action('get_group_list')
                for group in group_list:
                    if str(group['group_id']) == group_id:
                        return group['group_name']
            
            # 如果没有找到，返回格式化的群聊名称
            return f"群聊{group_id}"
            
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

            # self.jsonout(event.message_obj.message, f"收到的原始消息对象 - 群聊 {group_id} 用户 {user_id}")

            logger.debug(f"收到群聊消息: 群聊 {group_id}, 用户 {user_id}, 内容: {message_content[:50]}...")

            # 群聊白名单检查
            if not self._is_group_blacklisted(group_id):
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
            
            # 使用异步锁保护消息池访问
            async with self.message_pool_lock:
                # 遍历组件，遇到 Forward 组件时额外请求获取内部节点，并保留原始组件
                original_messages: List[Any] = []
                max_depth = int(self._get_config_value("FORWARD_MESSAGE_MAX_DEPTH", 3))
                
                for comp in event.get_messages():
                    if isinstance(comp, Comp.Forward):
                        # 处理合并转发消息，支持多层嵌套
                        try:
                            # 在此处进行一次API请求获取完整结构
                            resp = await event.bot.api.call_action('get_forward_msg', message_id=str(msg_id))
                            nested_structure = resp.get('messages', [])
                            # self.jsonout(nested_structure, f"用户 {user_id} 的原始消息组件")
                            
                            # 将完整结构传递给递归函数进行处理
                            nested_nodes = await self._process_forward_message_recursive(
                                nested_structure, timestamp, depth=0, max_depth=max_depth
                            )
                            original_messages.extend(nested_nodes)
                        except Exception as e:
                            logger.warning(f"处理合并转发消息失败: {e}")
                    else:
                        # self.jsonout(comp, f"用户 {user_id} 的comp")
                        original_messages.append(comp)
                self._add_message_to_pool(
                    group_id, user_id, timestamp,
                    str(msg_id) if msg_id else "",
                    original_messages
                )
                # self.jsonout(original_messages, f"用户 {user_id} 的消息组件")
                logger.debug(f"已将消息组件及转发节点添加到群聊 {group_id} 用户 {user_id} 的消息池")
            
            # 检查消息类型是否需要处理
            if not self._should_process_message_type(event):
                logger.debug(f"消息类型不需要处理，跳过检测: {message_content[:50]}...")
                return
            
            # 处理合并转发消息的内容提取
            extracted_text = message_content
            extracted_images = []
            
            # 如果包含合并转发消息，提取其中的文本和图片
            has_forward_message = any(isinstance(comp, Comp.Forward) for comp in event.get_messages())
            if has_forward_message and original_messages:
                logger.info(f"检测到合并转发消息，提取内容进行检测")
                extracted_text, extracted_images = self._extract_content_from_messages(original_messages)
                if extracted_text:
                    logger.debug(f"从合并转发消息中提取文本长度: {len(extracted_text)}")
                if extracted_images:
                    logger.debug(f"从合并转发消息中提取图片数量: {len(extracted_images)}")
            else:
                # 普通消息，提取图片URL
                for comp in event.get_messages():
                    if isinstance(comp, Comp.Image):
                        url = getattr(comp, 'url', None) or getattr(comp, 'file', None)
                        if url:
                            extracted_images.append(url)
            
            # 检查队列大小，避免积压过多
            max_queue_size = int(self._get_config_value("MAX_DETECTION_QUEUE_SIZE", 180))
            if self.detection_queue.qsize() >= max_queue_size:
                logger.warning(f"检测队列已满 ({self.detection_queue.qsize()})，跳过当前消息")
                return
            
            # 将检测任务加入队列：(群聊ID, 用户ID, 用户名, 消息内容, 发送时间, 图片URL列表, 事件对象)
            # 修改：预先提取图片URL列表，减少检测时的处理开销
            logger.debug(f"将消息加入检测队列: {extracted_text[:50]}...")
            detection_task = (group_id, user_id, user_name, extracted_text, timestamp, extracted_images, event)
            await self.detection_queue.put(detection_task)
            logger.debug(f"消息已加入队列，当前队列大小: {self.detection_queue.qsize()}")
                
        except Exception as e:
            logger.error(f"处理群聊消息时出错: {e}", exc_info=True)
    
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
            
            group_blacklist = self._get_config_value("BLACKLIST_GROUPS", [])
            config_status.append(f"群聊白名单: {len(group_blacklist)} 个群聊")
            
            user_whitelist = self._get_config_value("WHITELIST_USERS", [])
            config_status.append(f"用户白名单: {len(user_whitelist)} 个用户")
            
            # 检查模型配置
            text_model_api_key = self._get_config_value("TEXT_MODEL_API_KEY", "")
            config_status.append(f"文本模型API Key: {'已配置' if text_model_api_key else '❌ 未配置'}")
            
            vision_model_api_key = self._get_config_value("VISION_MODEL_API_KEY", "")
            config_status.append(f"视觉模型API Key: {'已配置' if vision_model_api_key else '❌ 未配置'}")
            
            # 检查批量处理配置
            batch_size = self._get_config_value("BATCH_PROCESS_SIZE", 3)
            batch_wait_time = self._get_config_value("BATCH_WAIT_TIME", 5.0)
            max_concurrent_ai = self._get_config_value("MAX_CONCURRENT_AI_CALLS", 3)
            config_status.append(f"批量处理配置: 批量大小={batch_size}, 等待时间={batch_wait_time}秒, AI并发限制={max_concurrent_ai}")
            
            # 检查当前群聊状态
            current_group = event.get_group_id()
            if current_group:
                is_group_blacklisted = self._is_group_blacklisted(current_group)
                config_status.append(f"当前群聊 {current_group}: {'✅ 在白名单中' if is_group_blacklisted else '❌ 不在白名单中'}")
            
            # 检查消息池状态
            total_groups = len(self.group_message_pools)
            total_users = sum(len(users) for users in self.group_message_pools.values())
            total_messages = sum(
                len(messages) for group in self.group_message_pools.values() 
                for messages in group.values()
            )
            config_status.append(f"消息池: {total_groups} 个群聊, {total_users} 个用户, {total_messages} 条消息记录")
            config_status.append(f"检测队列: {self.detection_queue.qsize()} 个待处理任务")
            config_status.append(f"正在处理的用户: {len(self.processing_users)} 个")
            
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
            current_group_id = event.get_group_id()
            test_messages = [
                {
                    "timestamp": time.time() - 60,
                    "message_id": "test_msg_1",
                    "recalled": False,
                    "original_messages": [Comp.Plain("这是测试消息1")]
                },
                {
                    "timestamp": time.time() - 30,
                    "message_id": "test_msg_2", 
                    "recalled": False,
                    "original_messages": [Comp.Plain("这是测试消息2")]
                }
            ]
            
            logger.info(f"开始测试转发功能到群聊: {admin_chat_id}")
            await self._forward_messages_as_merged(admin_chat_id, current_group_id, test_user_id, test_user_name, test_messages, event)
            
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
