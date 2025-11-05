import os
import sys
import argparse
import time
import gradio as gr
from openai import OpenAI
# from loguru import logger
# from utils.timer import Timer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

task_list = ['en_prompt', 'zh_prompt']

def parse_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--log_level', default='INFO', type=str,)
    # parser.add_argument('--model_path', default='local', type=str,)
    parser.add_argument('--llm_model', default='save_sft', type=str,) # qwen2-5-7b-instruct
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--task", type=str, default='origin')
    parser.add_argument("--devices", type=str, default='0')
    args = parser.parse_args()

    return args

def build_demo():

    def predict(message, history):
        # logger.info('predict')
        # Convert chat history to OpenAI format
        if args.task == 'en_prompt': # self prompt
            history_openai_format = [
                {"content": """
                I want you to act as an AI prompt engineer. You are expert at writing ChatGPT Prompts to get the best results.

                To create efficient prompts that yield high-quality responses, consider the following principles and strategies: \
                1. Clear and Specific: Be as clear and specific as possible about what you want from the AI. If you want a certain type of response, outline that in your prompt. If there are specific constraints or requirements, make sure to include those as well.
                2. Open-ended vs. Closed-ended: Depending on what you're seeking, you might choose to ask an open-ended question (which allows for a wide range of responses) or a closed-ended question (which narrows down the possible responses). Both have their uses, so choose according to your needs.
                3. Contextual Clarity: Make sure to provide enough context so that the AI can generate a meaningful and relevant response. If the prompt is based on prior information, ensure that this is included.
                4. Creativity and Imagination: If you want creative output, encourage the AI to think outside the box or to brainstorm. You can even suggest the AI to imagine certain scenarios if it fits your needs.

                There is a well-written prompt delimited by <> for your reference: <Your task is to be my brainstorming partner and provide creative ideas and suggestions for a given topic or problem. Your response should include original, unique, and relevant ideas that could help solve the problem or further explore the topic in an interesting way. Please note that your response should also take into account any specific requirements or constraints of the task.>

                Your task is to write an effective ChatGPT Prompt based on given keywords or to modify the given prompts. Answer in the same language as me.
                """,
                "role": "system"}
            ]
        elif args.task == 'ch_prompt': # self prompt
            history_openai_format = [
                {"content": """
                你是一个AI prompt工程师。编写ChatGPT prompt来获得最佳生成结果的专家。
    
                要创建产生高质量响应的高效提示，请考虑以下原则和策略： 
                1. 明确和具体：尽可能明确和具体你想从AI中得到什么。如果你想要某种类型的回复，在你的提示中列出它。如果有特定的限制或需求，确保也包括这些。 
                2. 开放式和封闭式：根据你想要问的问题，你可以选择问一个开放式的问题(这允许广泛的回答)或一个封闭式的问题(这缩小了可能的回答)。两者都有各自的用途，所以要根据自己的需要进行选择。 
                3. 情境清晰度：确保提供足够的情境，以便AI能够产生有意义且相关的回应。如果提示是基于先前的信息，请确保包含这些信息。 
                4. 创造力和想象力：如果你想要创造性的输出，那就鼓励AI跳出思维定式或进行头脑风暴。如果符合你的需求，你甚至可以建议AI想象某些场景。 
                
                这里有一个写得很好的prompt，以<>分隔供您参考：<你的任务是成为我的头脑风暴伙伴，并为给定的主题或问题提供创造性的想法和建议。你的回答应该包括原创的、独特的、相关的想法，这些想法可以帮助解决问题，或者以有趣的方式进一步探讨这个话题。请注意，您的回答还应考虑到任务的所有具体要求或限制。 
                
                你的任务是根据给定的关键词编写有效的ChatGPT prompt或修改给定的prompt。用和我一样的语言回答。
                """,
                "role": "system"}
            ]
        else: # origin
            history_openai_format = [{
                "role": "system",
                "content": "You are a great ai assistant."
            }]
        
        # history
        for human, assistant in history:
            if args.task == 'en_prompt': # self prompt
                history_openai_format.append({"role": "user", "content": "Please help me to write an effective ChatGPT Prompt based on the following keywords or prompt: {}".format(human)})
                history_openai_format.append({
                    "role": "assistant",
                    "content": assistant
                })
            elif args.task == 'ch_prompt': # self prompt
                history_openai_format.append({"role": "user", "content": "请根据以下关键词或prompt，帮我编写一个有效的ChatGPT prompt: {}".format(human)})
                history_openai_format.append({
                    "role": "assistant",
                    "content": assistant
                })
            else: # origin
                history_openai_format.append({"role": "user", "content": human})
                history_openai_format.append({
                    "role": "assistant",
                    "content": assistant
                })
        # message
        if args.task == 'en_prompt': # self prompt
            history_openai_format.append({"role": "user", "content": "Please help me to write an effective ChatGPT Prompt based on the following keywords or prompt: {}".format(message)})
        elif args.task == 'ch_prompt': # self prompt
            history_openai_format.append({"role": "user", "content": "请根据以下关键词或prompt，帮我编写一个有效的ChatGPT prompt: {}".format(message)})
        else: # origin
            history_openai_format.append({"role": "user", "content": message})
            
        # logger.info('history: {}'.format(history))
        # logger.info('history_openai_format: {}'.format(history_openai_format))
        # Create a chat completion request and send it to the API server
        stream = client.chat.completions.create(
            model=llm_model,  # Model name to use
            messages=history_openai_format,  # Chat history
            temperature=args.temperature,  # Temperature for text generation
            top_p=args.top_p, 
            max_tokens=256,
            stream=False, # stream=True,  # Stream response
            # extra_body={
            #     'repetition_penalty':
            #     1,
            #     'stop_token_ids': [
            #         int(id.strip()) for id in args.stop_token_ids.split(',')
            #         if id.strip()
            #     ] if args.stop_token_ids else []
            # }
            )

        # # Read and return generated text from response stream
        # partial_message = ""
        # for chunk in stream:
        #     partial_message += (chunk.choices[0].delta.content or "")
        #     yield partial_message

        answer = stream.choices[0].message.content
        # yield "", history+[[message,answer]]
        return "", history+[[message,answer]]
    

    def reset_state(msg,chatbot):
        # logger.info('reset_state')
        chatbot.clear()
        return gr.update(value=""), chatbot

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()

        if '_prompt' in args.task:
            msg = gr.Textbox(placeholder='What task u want me to prompt?')
        elif args.task == 'origin':
            msg = gr.Textbox(placeholder='Chat with me.')
        else:
            msg = gr.Textbox(placeholder='Chat with me.')
        
        clear = gr.Button("clear")

        msg.submit(predict, [msg, chatbot], [msg, chatbot], show_progress=True,) # queue=False,
        clear.click(reset_state, [msg, chatbot], [msg, chatbot], show_progress=True,) # queue=False,
    
    
    # http://172.28.4.10/
    demo.queue().launch(server_name=args.host, 
                        server_port=args.port,
                        share=False,debug=True)
    # return demo


if __name__ == "__main__":
    args = parse_args()
    llm_model = args.llm_model
    MODEL_PATH = llm_model
    client = OpenAI(api_key = "lyh-llm", base_url="http://localhost:8000/v1")
    try:
        res = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": "Are you ready?"}],
            temperature=0.9,
            max_tokens=256,
        ).choices[0].message.content
    except:
        if args.tp == 1:
            os.system("CUDA_VISIBLE_DEVICES={} python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --seed 0 \
            --served-model-name {} --model {} \
            --gpu-memory-utilization 0.95 \
            --dtype bfloat16 --api-key lyh-llm > vllm_server.log 2>&1 &".format(args.devices, llm_model, MODEL_PATH)) # --tensor-parallel-size 2 --max-model-len 10000
        elif args.tp > 1:
            os.system("CUDA_VISIBLE_DEVICES={} python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --seed 0 \
            --served-model-name {} --model {} \
            --gpu-memory-utilization 0.9 --tensor-parallel-size {}\
            --dtype bfloat16 --api-key lyh-llm > vllm_server.log 2>&1 &".format(args.devices, llm_model, MODEL_PATH), args.tp) # --tensor-parallel-size 2 --max-model-len 10000
        else:
            raise ValueError('Start server error')
    # llm warmup
    # timer = Timer()
    # logger.info("Warm up:")
    time.sleep(10)
    query = "Are you ready?"
    for i in range(30):
        try:
            # timer.refresh()
            res = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": query}],
                temperature=0.9,
                max_tokens=256,
            ).choices[0].message.content
            # logger.debug(f'Warp up finished:{res}')
            # timer.check_time()
            break
        except:
            # logger.info('llm server not started, waiting...')
            time.sleep(10)
    os.system("cat ./vllm_server.log")

    build_demo()
    