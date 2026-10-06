"""
@Author:shkstart
@Desc:  演示 LangChain 异步批量调用 abatch 的非阻塞特性
"""
# ========== 1. 导入需要的模块 ==========

import asyncio
# asyncio 是 Python 自带的异步编程库。
# 它提供事件循环、协程调度、create_task、gather、sleep 等异步能力。

import os
# os 用来读取操作系统环境变量，例如 os.getenv("XXX")。

from langchain.chat_models import init_chat_model
# LangChain 提供的统一聊天模型初始化函数。
# 不同厂商的模型，都可以通过它来创建，比如 OpenAI、DeepSeek 等。

from dotenv import load_dotenv
# 从 .env 文件中读取环境变量，避免把 API Key 直接写死在代码里。

import time
# time 用来计时。
# 这里主要用 time.perf_counter() 来统计程序运行耗时。


# ========== 2. 加载环境变量 ==========

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

# 从环境变量中读取 DeepSeek 的 API Key 和 Base URL。
# 注意：这些变量名必须和 .env 文件中写的名字完全一致。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")


# ========== 3. 初始化 LangChain 聊天模型 ==========

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
)


# ========== 4. 定义协程函数 ==========

# 涉及 python 知识点：协程
# async def 定义的是协程函数。直接调用它，不会立刻执行里面的代码，而是返回一个“协程对象”。
# 协程对象必须交给事件循环运行，例如通过 asyncio.run()、await、create_task() 等。
#
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
    # model.abatch(questions)：
    # 这是 LangChain 的异步批量调用方法，会返回一个协程对象。
    # 它表示“将来会完成”的批量请求，但此时还没有真正开始等待结果。
    #
    # asyncio.create_task(...)：
    # 把上面返回的协程对象包装成一个 Task，并注册到事件循环中。
    # 这一步不会阻塞当前代码，而是立即返回 batch_task。
    # 之后主程序可以继续往下执行，后台任务会由事件循环调度。

    # 2. 在等待批量处理的同时，执行其他任务
    print(">>> 批量任务已在后台运行，主程序继续执行...")
    for i in range(3):
        # 关键修改：使用 asyncio.sleep 允许后台任务获取 CPU 时间片进行网络请求
        await asyncio.sleep(1)      # 异步休眠 1 秒
        # 注意：这里不能用 time.sleep(1)。
        # time.sleep(1) 是同步阻塞，会卡住整个事件循环，后台 batch_task 无法推进。
        #
        # await asyncio.sleep(1) 会让出事件循环控制权，

        # 打印当前主程序执行进度和已耗时。
        print(
            f">>> 正在执行第{i + 1}个任务... "
            f"(已耗时 {time.perf_counter() - start_time:.2f}s)"
        )

    # 3. 等待批量处理结果
    print(">>> 其他任务已完成，现在获取后台批量任务的结果...")
    # 此时 batch_task 可能已经完成，或者我们在这里等待它完成
    responses = await batch_task

    # 记录结束时间
    end_time = time.perf_counter()

    # 遍历批量请求返回的结果
    for response in responses:
        content = response.content if hasattr(response, 'content') else str(response)
        # hasattr(response, 'content') 用来安全检查：
        # 如果 response 有 content 属性，就使用 response.content；
        # 否则直接转成字符串，避免报错。
        print(f">>> 响应内容: {content}")

    print(f"=== 总运行耗时: {end_time - start_time:.2f}s ===")


# ========== 5. 主协程 ==========

async def main():
    """主函数"""
    # main 是一个协程函数。
    # 这里 await demo_async_batch()，表示等待 demo_async_batch 执行完成。
    await demo_async_batch()


# ========== 6. 程序入口 ==========

if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(main()) 做三件事：
    # 1. 创建一个新的事件循环。
    # 2. 运行 main() 协程，直到它完成。
    # 3. 关闭事件循环。
    #
    # 注意：asyncio.run() 通常只能在程序入口调用一次，
    # 不要嵌套调用，也不要在已经有事件循环的环境里反复调用。
