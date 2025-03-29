import json5
import time
from openai import OpenAI
from configs import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL


def extrack_json_data(text: str):
    # 提取最后一个json文件
    if "```json" in text:
        start_idx = text.rfind("```json") + 7
    elif text.count("```") >= 2:
        start_idx = text.rfind("```") + 3
    else:
        return text
    end_idx = text.find("```", start_idx)
    json_data = text[start_idx:end_idx]

    try:
        json_data = json5.loads(json_data)
    except:
        json_data = eval(json_data)

    return json_data


system_prompt = """
你是一名中文教授，你的任务是总结用户输入的内容并提取其中核心的关键词（少于10个）。

必须返回JSON格式：
```json
{"summary":"<总结内容>", "keywords":["<关键词1>", "<关键词2>", ...]}
```
""".strip()


def summary(text, model=SILICONFLOW_MODEL):
    """总结段落

    Args:
        text (str): 输入的文本
        model (str, optional): 模型名字. Defaults to "Qwen/Qwen2.5-7B-Instruct".

    Returns:
        dict: 返回数据
    """
    keywords = text
    retry_times = 10
    while retry_times > 0:
        try:
            client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            content = extrack_json_data(content)
            keywords = ",".join(content["keywords"])
            break
        except Exception as e:
            retry_times -= 1
            print(e, f"[{10-retry_times+1}/{retry_times}] waiting for 5s.")
            time.sleep(5)

    return text, keywords
