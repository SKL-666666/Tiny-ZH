"""下载 minimind-3 权重，用于对比测试。"""
import os
import sys

from modelscope.hub.file_download import model_file_download

OUT = "models/minimind-3"

FILES = ["model.safetensors", "config.json", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json",
         "chat_template.jinja", "generation_config.json"]

os.makedirs(OUT, exist_ok=True)
for f in FILES:
    try:
        p = model_file_download(model_id="gongjy/minimind-3",
                                file_path=f, local_dir=OUT)
        size = os.path.getsize(p)
        print(f"  {f:<28} {size/1e6:>8.2f} MB")
    except Exception as e:
        print(f"  {f:<28} FAIL {str(e)[:60]}")
