from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import json
import argparse
from configparser import ConfigParser
from datetime import datetime
import sys

# json読み込み
def load_models():
    global models
    json_open = open("./models.json", "r")
    models = json.load(json_open)
load_models()

# ini読み込み
def load_config():
    global config
    ini = ConfigParser()
    ini.read("./settings.ini", encoding="utf-8")
    config = ini["DEFAULT"]
load_config()

# Stable Diffusion
def stableDiffusion(prompt, model_id, n_prompt):
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                                   torch_dtype=torch.float16, 
                                                   revision="fp16",
                                                   scheduler=scheduler,
                                                   custom_pipeline="lpw_stable_diffusion"
                                                   )
    
    # メモリ系最適化
    pipe = pipe.to("cuda")
    if eval(config["enable_attention_slicing"]):
        pipe.enable_attention_slicing()

    # filter無効化
    def null_safety(images, **kwargs):
        return images, False
    if eval(config["enable_nsfw"]) and not "stable-diffusion-2" in model_id:
        pipe.safety_checker = null_safety


    if n_prompt != "":
        image = pipe(prompt, width=512, height=512, negative_prompt=n_prompt, max_embeddings_multiples=int(config["max_embeddings_multiples"])).images[0]
    else:
        image = pipe(prompt, width=512, height=512, max_embeddings_multiples=int(config["max_embeddings_multiples"])).images[0]
    return image

# 読み込み
load_models()
load_config()
args = sys.argv[1:]

# model listを表示
if len(args) != 0 and args[0] == "--show-models":
    msg = "== model list =="
    for model in models:
        msg += "\n" + model["model"]
    print(msg)
    exit(0)

# 引数を解析
try:
    parser = argparse.ArgumentParser() 
    parser.add_argument("prompt", nargs=1, help="Prompt to generate image.")
    parser.add_argument("--model", "-m", type=str, help="Model name.", default=config["default_model"])
    parser.add_argument("--times", "-t", type=int, help="Export Count.", default=1)
    parser.add_argument("--negative", "-n", type=str, help="Negative prompt.", default="")
    parsed_args = parser.parse_args(args)
except:
    print("Argument error.")
    exit(0)
# プロンプト
if hasattr(parsed_args, "prompt") and parsed_args.prompt is not None:
    prompt = parsed_args.prompt
else:
    print("Prompt not found.")
    exit(0)
if hasattr(parsed_args, "negative") and parsed_args.negative is not None:
    n_prompt = parsed_args.negative

# モデルIDを決定
model_id = ""
if hasattr(parsed_args, "model") and parsed_args.model is not None:
    exist = False
    for model in models:
        if model["model"] == parsed_args.model:
            model_id = model["path"]
            exist = True
            break
    if not exist:
        print("Model not found.")
        exit(0)
    
# 生成回数
times = parsed_args.times

# 生成
for i in range(times):
    image = stableDiffusion(prompt, model_id, n_prompt)
    image.save(f'{config["output_dir"]}/{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

