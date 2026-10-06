"""
@Author:shkstart
@Desc: 
"""
import asyncio                                          # Python 自带的异步编程库，用来写 async / await 代码
import os                                               # 读取操作系统环境变量
from langchain.chat_models import init_chat_model       # LangChain 提供的统一模型初始化函数
from dotenv import load_dotenv                          # 从 .env 文件加载环境变量
import time                                             # 计时用

# # 从.env文件中加载环境变量
# load_dotenv(override=True)          # override=True 表示：如果系统里已经有同名环境变量，也用 .env 里的值覆盖它。默认是 False。
#
# CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY")
# CLOSEAI_BASE_URL = os.getenv("CLOSEAI_BASE_URL")
#
# # 初始化 LangChain 聊天模型
# model = init_chat_model(
#     model="openai:gpt-5.4-mini",
#     api_key=CLOSEAI_API_KEY,
#     base_url=CLOSEAI_BASE_URL
# )

# 1、读取.env配置文件中的信息。相关的环境变量以.env文件中的优先
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 2、模型的初始化
model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
)

# 涉及 python 知识点：协程
'''

'''
# async def 定义的是协程函数。直接调用它，不会立刻执行里面的代码，而是返回一个“协程对象”。协程对象必须交给事件循环运行。
# await 表示“等待某个异步操作完成”。但它和普通等待不一样：当前协程暂停，事件循环可以去执行其他任务。
async def demo_async_batch():
    """演示异步批量的非阻塞特性"""
    print("=== 演示：abatch 的异步（非阻塞）效果 ===")
    start_time = time.perf_counter()  # 记录开始时间

    print("程序开始...")

    # 准备批量输入
    questions = ["用一句话说明深度学习与传统机器学习的区别", "中国首都在哪里？"]

    # 1. 发起异步批量请求
    # 关键修改：使用 create_task 让协程立即在后台执行
    print(">>> 发起异步批量调用 (abatch)...")
    batch_task = asyncio.create_task(model.abatch(questions))

    # 2. 在等待批量处理的同时，执行其他任务
    print(">>> 批量任务已在后台运行，主程序继续执行...")
    for i in range(3):
        # 关键修改：使用 asyncio.sleep 允许后台任务获取 CPU 时间片进行网络请求
        await asyncio.sleep(1)
        print(f">>> 正在执行第{i + 1}个任务... (已耗时 {time.perf_counter() - start_time:.2f}s)")

    # 3. 等待批量处理结果
    print(">>> 其他任务已完成，现在获取后台批量任务的结果...")
    # 此时 batch_task 可能已经完成，或者我们在这里等待它完成
    responses = await batch_task

    end_time = time.perf_counter()

    for response in responses:
        content = response.content if hasattr(response, 'content') else str(response)
        print(f">>> 响应内容: {content}")

    print(f"=== 总运行耗时: {end_time - start_time:.2f}s ===")


async def main():
    """主函数"""
    await demo_async_batch()


if __name__ == "__main__":
    asyncio.run(main())     # 1.创建一个新的事件循环。 2.运行 main() 协程。 3.等它完成后关闭事件循环。
