#!/usr/bin/env python3
"""
AstrBot 防推销插件测试脚本
用于验证插件基本结构和配置的正确性
"""

import json
import os
import sys

def test_plugin_structure():
    """测试插件目录结构"""
    print("🔍 检查插件目录结构...")
    
    required_files = [
        "main.py",
        "metadata.yaml", 
        "_conf_schema.json",
        "requirements.txt",
        "README.md"
    ]
    
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(plugin_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            print(f"  ✅ {file}")
    
    if missing_files:
        print(f"  ❌ 缺少文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 插件目录结构检查通过")
    return True

def test_metadata():
    """测试metadata.yaml文件"""
    print("\n🔍 检查metadata.yaml...")
    
    try:
        with open("metadata.yaml", "r", encoding="utf-8") as f:
            content = f.read()
        
        required_fields = ["name:", "desc:", "version:", "author:", "repo:"]
        missing_fields = []
        
        for field in required_fields:
            if field in content:
                # 提取字段值
                lines = content.split('\n')
                for line in lines:
                    if line.strip().startswith(field):
                        value = line.split(':', 1)[1].strip().split('#')[0].strip()
                        print(f"  ✅ {field[:-1]}: {value}")
                        break
            else:
                missing_fields.append(field[:-1])
        
        if missing_fields:
            print(f"  ❌ 缺少字段: {', '.join(missing_fields)}")
            return False
        
        print("✅ metadata.yaml检查通过")
        return True
        
    except Exception as e:
        print(f"  ❌ metadata.yaml解析错误: {e}")
        return False

def test_config_schema():
    """测试配置模式文件"""
    print("\n🔍 检查_conf_schema.json...")
    
    try:
        with open("_conf_schema.json", "r", encoding="utf-8") as f:
            schema = json.load(f)
        
        expected_configs = [
            "LAST_TIME",
            "ADMIN_CHAT_ID", 
            "SPAM_ALERT_MESSAGE",
            "MUTE_DURATION",
            "CONTEXT_MESSAGE_COUNT",
            "WHITELIST_USERS",
            "WHITELIST_GROUPS",
            "LLM_SYSTEM_PROMPT",
            "TEXT_MODEL_ID",
            "TEXT_MODEL_BASE_URL", 
            "TEXT_MODEL_API_KEY",
            "VISION_MODEL_ID",
            "VISION_MODEL_BASE_URL",
            "VISION_MODEL_API_KEY",
            "MODEL_TIMEOUT"
        ]
        
        missing_configs = []
        
        for config in expected_configs:
            if config not in schema:
                missing_configs.append(config)
            else:
                config_info = schema[config]
                print(f"  ✅ {config}: {config_info.get('type', 'unknown')} - {config_info.get('description', 'No description')}")
        
        if missing_configs:
            print(f"  ❌ 缺少配置项: {', '.join(missing_configs)}")
            return False
        
        print("✅ _conf_schema.json检查通过")
        return True
        
    except Exception as e:
        print(f"  ❌ _conf_schema.json解析错误: {e}")
        return False

def test_main_structure():
    """测试main.py基本结构"""
    print("\n🔍 检查main.py结构...")
    
    try:
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        required_patterns = [
            "@register(",
            "class SpamDetectorPlugin(Star):",
            "def __init__(self, context: Context, config: AstrBotConfig):",
            "@filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)",
            "async def on_group_message(self, event: AstrMessageEvent):",
            "_is_user_whitelisted",
            "_is_group_whitelisted",
            "_is_spam_message",
            "_handle_spam_message",
            "from astrbot.api import logger, AstrBotConfig"
        ]
        
        missing_patterns = []
        
        for pattern in required_patterns:
            if pattern in content:
                print(f"  ✅ 找到: {pattern}")
            else:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            print(f"  ❌ 缺少代码结构: {', '.join(missing_patterns)}")
            return False
        
        print("✅ main.py结构检查通过")
        return True
        
    except Exception as e:
        print(f"  ❌ main.py读取错误: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 AstrBot 防推销插件测试开始")
    print("=" * 50)
    
    tests = [
        test_plugin_structure,
        test_metadata,
        test_config_schema,
        test_main_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！插件结构正确。")
        print("\n📝 下一步:")
        print("1. 将插件目录复制到 AstrBot 的 data/plugins/ 目录")
        print("2. 在 AstrBot 管理面板配置插件参数")
        print("3. 重载插件并测试功能")
        return 0
    else:
        print(f"❌ {total - passed} 个测试失败，请检查并修复问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
