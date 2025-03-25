import requests

def get_current_weather(location: str, unit: str = "celsius"):
    """获取指定城市的当前天气（使用 OpenWeatherMap API）"""
    API_KEY = "d49e03bbdac72ab2ddcac7c3145546ee"  # 替换成你的 OpenWeatherMap API Key
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": location,
        "appid": API_KEY,
        "units": "metric" if unit == "celsius" else "imperial",
    }
    print(params)
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data["cod"] != 200:
            return f"Error: {data['message']}"
        
        weather = {
            "location": location,
            "temperature": data["main"]["temp"],
            "unit": unit,
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
        }
        print(weather)
        return json.dumps(weather)
    except Exception as e:
        return f"Error fetching weather: {str(e)}"
    
from openai import OpenAI
import json

client = OpenAI(
    api_key="sk-",  # 替换成你的 DeepSeek API Key
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location. If input in Chinese, please translate it to English.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco, CA.  ",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

# 用户输入
messages = [{"role": "user", "content": "杭州今天的天气?"}]
print(f"User>\t {messages[0]['content']}")

# Step 1: 发送消息，让模型决定是否调用天气函数
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
)
assistant_message = response.choices[0].message
messages.append({
    "role": "assistant",
    "content": assistant_message.content,
    "tool_calls": assistant_message.tool_calls,
})

# Step 2: 检查是否有天气函数调用
if assistant_message.tool_calls:
    tool_call = assistant_message.tool_calls[0]
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    # 调用天气 API
    weather_data = get_current_weather(**args)
    
    # 添加 tool 消息（必须包含 tool_call_id）
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": func_name,
        "content": weather_data,
    })

    # Step 3: 发送最终请求，让模型生成自然语言回复
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    print(f"Model>\t {final_response.choices[0].message.content}")
else:
    print("No function was called.")
