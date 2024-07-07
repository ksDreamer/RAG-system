import os 
import json
import requests
from dashscope import Generation
from http import HTTPStatus


sys_content = "You are a helpful assistant. You must seek answers within the provided context first \
and output answers based on the context. When you are unable to find the answer within the \
context, you must use your own knowledge base to answer the question. You are not allowed \
to refuse to answer. If you are forced to answer without being able to find the answer, \
you need to indicate at the end of your response, in parentheses, that this answer was not \
found in the context. For questions with a definitive answer, provide the key answer directly \
without lengthy explanations. The output should be in Chinese."

# 使用通义千问获取答案
# 这里使用了环境变量来获取Key
# 需设置如下的环境变量
# export QIANFAN_ACCESS_KEY="xxxx"
# export QIANFAN_SECRET_KEY="xxxx"
def generate_answer_qwen(query, context):
    
    messages = [
            {
                "role": "system",
                "content": sys_content
            },
            {
                "role": "user",
                "content": f"问题: {query} 上下文: {context}"
            }
        ]
    
    
    responses = Generation.call(model="qwen-max",
                                messages=messages,
                                result_format='message',  # 设置输出为'message'格式
                                stream=True,  # 设置输出方式为流式输出
                                incremental_output=True  # 增量式流式输出
                                )
    #print(messages)
    answer = ''
    for response in responses:
        if response.status_code == HTTPStatus.OK:

            answer += response.output.choices[0]['message']['content']

        else:
            answer = f'Request ERROR: Request id: {response.request_id}, Status code: {response.status_code}, \
                error code: {response.code}, error message: {response.message}'
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            break
    return answer


def generate_answer_gpt(query, context):
    # 定义 API 密钥
    # 请在自己电脑上配置 OpenAI API KEY 环境变量
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    API_URL = os.environ.get('API_URL') # OpenAI或者你选择的服务商与 HTTP Post 相关的 URL，例如 "https://api2.aigcbest.top/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": sys_content
            },
            {
                "role": "user",
                "content": f"问题: {query} 上下文: {context}"
            }
        ]
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(data)) # 从官方 API 文档获得更多信息
    response_json = response.json()
    answer = response_json['choices'][0]['message']['content']
    return answer


def generate_answer(query, context, model="gpt"):       
    if model == "qwen":
        return generate_answer_qwen(query, context)
    elif model == "gpt":
        return generate_answer_gpt(query, context)
    else:
        return "Unsupported model"
