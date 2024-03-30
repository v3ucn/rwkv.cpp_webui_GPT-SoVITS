import argparse
import time
import sampling
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from tokenizer_util import add_tokenizer_argument, get_tokenizer
from rwkv_cpp import rwkv_world_tokenizer
from typing import List
import gradio as gr
import time
import sampling

import requests
import json

from pydub import AudioSegment
from pydub.playback import play


title = "RWKV-5-H-World"

model = None

tokenizer_decode = None

tokenizer_encode = None


def read_now(text):

    if text == "":
        return "请输入要播放的文本"

    data = json.dumps({"text":text})#转换为json字符串
    headers = {"Content-Type":"application/json"}#指定提交的是json
    r = requests.post("http://localhost:9880/tts_to_audio/",data=data,headers=headers)

    with open('./output_audio.wav', 'wb') as audio_file:
        audio_file.write(r.content)
    sound = AudioSegment.from_file("./output_audio.wav", format="wav")
    play(sound)


def load_model(model_name):

    # Load the model accordingly

    global model,tokenizer_decode,tokenizer_encode
    
    library = rwkv_cpp_shared_library.load_rwkv_shared_library()
    print(f'System info: {library.rwkv_get_system_info_string()}')

    print('Loading RWKV model')
    model = rwkv_cpp_model.RWKVModel(library,model_name)

    tokenizer_decode, tokenizer_encode = get_tokenizer('auto', model.n_vocab)

    return "加载成功"




def evaluate(
    ctx,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty = 0.1,
    countPenalty = 0.1,
):
  prompt_tokens = tokenizer_encode(ctx)

  prompt_token_count = len(prompt_tokens)

  init_logits, init_state = model.eval_sequence_in_chunks(prompt_tokens, None, None, None, use_numpy=True)

  logits, state = init_logits.copy(), init_state.copy()

  out_str = ''

  occurrence = {}

  for i in range(token_count):
    for n in occurrence:
      logits[n] -= (presencePenalty + occurrence[n] * countPenalty)

    token: int = sampling.sample_logits(logits, temperature, top_p)

    tk = tokenizer_decode([token])
    #print(tokenizer_decode([token]), end='', flush=True)
    out_str+=tk
    yield out_str

    for xxx in occurrence:
      occurrence[xxx] *= 0.996

    if token not in occurrence:
      occurrence[token] = 1
    else:
      occurrence[token] += 1

    logits, state = model.eval(token, state, state, logits, use_numpy=True)

def main():

    # Gradio blocks
    with gr.Blocks(title=title) as demo:
        gr.HTML(f"<div style=\"text-align: center;\">\n<h1>RWKV-5 World v2 - {title}</h1>\n</div>")

        gr.HTML(f"<div style=\"text-align: center;\">\n<h1>模型量化说明</h1>\n模型量化精度从高到低排列顺序是：fp16>int8>int4，量化的精度越低，模型的大小和推理所需的显存就越小，但模型的性能也会越差。\n这里量化后的模型都是ggml格式，默认使用Q5_1的量化版本，它的体积只有全精度的一半，但是性能会下降。另外提供fp16的量化版本，精度损失最小，性能接近原版，但是速度要比Q5_1的量化版本要慢。</div>")

            
        with gr.Tab("RWKV-5-h"):
            gr.Markdown(f"This is RWKV-5-h ")

            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(label="模型列表", choices=["./rwkv-5-h-world-7B-Q5_1.bin", "./rwkv-5-h-world-7B-fp16.bin"], value="./rwkv-5-h-world-7B-Q5_1.bin", interactive=True)
          

                    b_load = gr.Button("加载模型", variant="primary")
                    b_output = gr.Textbox(label="加载结果")

            b_load.click(load_model, [model_dropdown], [b_output])


            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(lines=2, label="Prompt", value="""边儿上还有两条腿，修长、结实，光滑得出奇，潜伏着
    媚人的活力。他紧张得脊梁都皱了起来。但他不动声色。""")
                    token_count = gr.Slider(10, 500, label="Max Tokens", step=10, value=200)
                    temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                    top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                    presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=1)
                    count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button("开始推理", variant="primary")
                        clear = gr.Button("Clear", variant="secondary")
                    output = gr.Textbox(label="Output", lines=5)
                    read_b = gr.Button("开始朗读", variant="primary")
            data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], label="Example Instructions", headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
            submit.click(evaluate, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
            clear.click(lambda: None, [], [output])
            data.click(lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])

            read_b.click(read_now,[output],[])

    demo.queue()
    demo.launch(server_name="0.0.0.0",inbrowser=True)

if __name__ == '__main__':
    main()
