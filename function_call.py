                                        
from openai import OpenAI
import json
import math

# 1. 定义数学计算函数
def math_operation(operation: str, num1: float, num2: float = None):
    if operation == "add":
        return str(num1 + num2)
    elif operation == "subtract":
        return str(num1 - num2)
    elif operation == "multiply":
        return str(num1 * num2)
    elif operation == "divide":
        if num2 == 0:
            return "Error: Division by zero"
        return str(num1 / num2)
    elif operation == "square_root":
        return str(math.sqrt(num1))
    else:
        return "Error: Unknown operation"

# 2. 配置 DeepSeek API
client = OpenAI(
    api_key="sk-36ff43f5ce",  # 替换为你的 DeepSeek API Key
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "math_operation",
            "description": "Perform a mathematical calculation (add/subtract/multiply/divide/square_root)",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "square_root"],
                        "description": "The type of mathematical operation",
                    },
                    "num1": {
                        "type": "number",
                        "description": "The first number",
                    },
                    "num2": {
                        "type": "number",
                        "description": "The second number (required for add/subtract/multiply/divide)",
                    },
                },
                "required": ["operation", "num1"],
            },
        },
    }
]

# 初始用户消息
messages = [{"role": "user", "content": " 345 的平方根 "}]
print(f"User>\t {messages[0]['content']}")

# Step 1: 发送消息，让模型决定是否调用函数
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
)
assistant_message = response.choices[0].message
print(f"Assistant>\t {assistant_message}")
messages.append({"role": "assistant", "content": assistant_message.content, "tool_calls": assistant_message.tool_calls})

# Step 2: 检查是否有函数调用
if assistant_message.tool_calls:
    tool_call = assistant_message.tool_calls[0]
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    # 执行函数
    result = math_operation(**args)
    
    # 添加 tool 消息（必须包含 tool_call_id）
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": func_name,
        "content": result,
    })

    # Step 3: 发送最终请求，让模型生成自然语言回复
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    print(f"Model>\t {final_response.choices[0].message.content}")
else:
    print("No function was called.")
