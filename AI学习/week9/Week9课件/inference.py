import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm import load, generate
def predict(messages, model, tokenizer):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # # todo MLX的分词器不直接支持apply_chat_template，需要手动构建prompt
    # prompt = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # # 使用MLX的generate方法（参数与Transformers不同）
    # response = generate(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prompt=prompt,
    #     max_tokens=2048  # 对应max_new_tokens
    # )


    return response


# 加载原下载路径的tokenizer和model
# 使用实际下载的目录名，并根据设备选择合适的dtype
model_dir = "./models/Qwen/Qwen3-1___7B"
if torch.cuda.is_available():
    load_dtype = torch.float16
else:
    load_dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=load_dtype)

# todo mac单独支持的框架版本
# model_dir = "./models/Qwen/Qwen3-1___7B-MLX-8bit"
# model, tokenizer = load(
#     model_dir  # 明确指定使用Apple的GPU加速
# )
test_texts = {
    'instruction': "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
    'input': "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

response = predict(messages, model, tokenizer)
print(response)