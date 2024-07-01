# 从零实现Llama3
在这个文件中，我从零开始实现了Llama3，一次一个张量和矩阵乘法。
<br>
此外，我将直接从Meta提供的Llama3模型文件中加载张量，在运行此文件之前，您需要下载权重。
这是下载权重的官方链接：https://llama.meta.com/llama-downloads/

<div>
    <img src="images/archi.png"/>
</div>

## 分词器
我不会实现BPE分词器（但Andrej Karpathy有一个非常简洁的实现）
<br>
他的实现链接：https://github.com/karpathy/minbpe

<div>
    <img src="images/karpathyminbpe.png" width="600"/>
</div>



```python
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt

tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
special_tokens = [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",  # 结束标记
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

tokenizer.decode(tokenizer.encode("hello world!"))
```




    'hello world!'



## 读取模型文件
通常，读取模型文件取决于模型类的编写方式和其中的变量名。
<br>
但由于我们是从零实现Llama3，我们将一次读取一个张量。
<div>
    <img src="images/model.png" width="600"/>
</div>


```python
model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
print(json.dumps(list(model.keys())[:20], indent=4))
```

    [
        "tok_embeddings.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w3.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.1.attention.wq.weight",
        "layers.1.attention.wk.weight",
        "layers.1.attention.wv.weight",
        "layers.1.attention.wo.weight",
        "layers.1.feed_forward.w1.weight",
        "layers.1.feed_forward.w3.weight",
        "layers.1.feed_forward.w2.weight",
        "layers.1.attention_norm.weight",
        "layers.1.ffn_norm.weight",
        "layers.2.attention.wq.weight"
    ]



```python
with open("Meta-Llama-3-8B/params.json", "r") as f:
    config = json.load(f)
config
```




    {'dim': 4096,
     'n_layers': 32,
     'n_heads': 32,
     'n_kv_heads': 8,
     'vocab_size': 128256,
     'multiple_of': 1024,
     'ffn_dim_multiplier': 1.3,
     'norm_eps': 1e-05,
     'rope_theta': 500000.0}



## 我们使用这个配置来推断模型的细节，例如
1. 该模型有32个Transformer层
2. 每个多头注意力块有32个头
3. 词汇表大小等


```python
dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])
```

## 将文本转换为标记
这里我们使用tiktoken（我认为是一个OpenAI的库）作为分词器
<div>
    <img src="images/tokens.png" width="600"/>
</div>


```python
prompt = "the answer to the ultimate question of life, the universe, and everything is "
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)
tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
print(prompt_split_as_tokens)
```

    [128000, 1820, 4320, 311, 279, 17139, 3488, 315, 2324, 11, 279, 15861, 11, 323, 4395, 374, 220]
    ['', 'the', ' answer', ' to', ' the', ' ultimate', ' question', ' of', ' life', ',', ' the', ' universe', ',', ' and', ' everything', ' is', ' ']


## 将标记转换为其嵌入
对不起，这是代码库中唯一使用内置神经网络模块的部分
<br>
无论如何，我们的[17x1]标记现在变成了[17x4096]，即17个嵌入（每个标记一个），长度为4096
<br>
<br>
注意：请跟踪形状，这会让理解一切变得更容易

<div>
    <img src="images/embeddings.png" width="600"/>
</div>


```python
embedding_layer = torch.nn.Embedding(vocab_size, dim)
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
token_embeddings_unnormalized.shape
```




    torch.Size([17, 4096])



## 我们然后使用RMS规范化嵌入
请注意，在此步骤之后形状不会改变，值只是被规范化了
<br>
需要注意的是，我们需要norm_eps（来自配置）因为我们不希望RMS意外设置为0并除以0
<br>
公式如下：
<div>
    <img src="images/rms.png" width="600"/>
</div>


```python
# def rms_norm(tensor, norm_weights):
#     rms = (tensor.pow(2).mean(-1, keepdim=True) + norm_eps)**0.5
#     return tensor * (norm_weights / rms)
def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights
```

# 构建第一个Transformer层

### 规范化
你会看到我从模型字典中访问layer.0（这是第一层）
<br>
无论如何，规范化后我们的形状仍然是[17x4096]，与嵌入相同，但已规范化

<div>
    <img src="images/norm.png" width="600"/>
</div>


```python
token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"])
token_embeddings.shape
```




    torch.Size([17, 4096])



### 从头实现注意力机制
让我们加载Transformer第一层的注意力头
<div>
    <img src="images/qkv.png" width="600"/>
</div>

<br>

&gt; 当我们从模型中加载查询、键、值和输出向量时，我们注意到形状分别为[4096x4096]，[1024x4096]，[1024x4096]，[4096x4096]
<br>
&gt; 乍一看这有点奇怪，因为理想情况下我们希望每个头分别有q、k、v和o
<br>
&gt; 代码的作者将它们捆绑在一起，因为这可以帮助并行化注意力头的乘法。
<br>
&gt; 我将解包所有内容... 


```python
print(
    model["layers.0.attention.wq.weight"].shape,
    model["layers.0.attention.wk.weight"].shape,
    model["layers.0.attention.wv.weight"].shape,
    model["layers.0.attention.wo.weight"].shape
)
```

    torch.Size([4096, 4096]) torch.Size([1024, 4096]) torch.Size([1024, 4096]) torch.Size([4096, 4096])


### 解包查询
在接下来的部分中，我们将从多个注意力头中解包查询，结果形状为[32x128x4096]
<br><br>
这里，32是Llama3中的注意力头数量，128是查询向量的大小，4096是标记嵌入的大小


```python
q_layer0 = model["layers.0.attention.wq.weight"]
head_dim = q_layer0.shape[0] // n_heads
q_layer0 = q_layer0.view(n_heads, head_dim,
