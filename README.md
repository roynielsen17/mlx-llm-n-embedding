# README.md

Loosely based on the [kibotu/mlx-llm-server-mac-m-series](https://github.com/kibotu/mlx-llm-server-mac-m-series) repository.

Put together for an assignment, 

* run_llm.py
* run_embedding_model.py

are servers that provide services leverage Apples MLX AI 
capabilities that run on Apple M-series hardware.

The servers are in an Alpha state, not extensively tested.

I would welcome contributions to improve the servers towards
making them production capable.

Working towards tests, requirements.txt, and a more 
pythonic repo structure. 

Before running the rag pipeline, on your M-series Mac, run:

``` zsh
 python run_embedding_model.py \
--model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
--port 8898 --host 127.0.0.1
```

(feel free to run a smaller qwen model
``` zsh
python run_llm.py \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --host 127.0.0.1 \
  --port 8899
```

then you can run the rag model:

``` zsh
python rag_pipeline.py
```

For now, you will need to manually install dependancies.

