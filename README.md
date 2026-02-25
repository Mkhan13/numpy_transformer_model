# Numpy Transformer Model
This project implements an encoder only and decoder only Transformer model from scratch using only NumPy. The implementation focuses on understanding the core architecture of Transformers by building the model components manually without relying on high-level deep learning frameworks.

## Architecture Details

## Key Components
```_generate_positional_encoding:``` Generates sine/cosine-based encodings to inject position information into the token embeddings.

```_multi_head_attention:``` Implements scaled dot-product attention with softmax normalization.

```_feed_forward:``` A 2-layer MLP with ReLU activation.

```_layer_norm:``` Applies mean-variance normalization across the feature dimension.

```forward:``` Runs the input through the full transformer block (attention + FFN)

## How to Run
In a new python file, run the following code:
```
import numpy as np
from transformer import Transformer

x = np.random.rand(2, 512, 768)  # test input (batch_size=2)
model = Transformer()
output = model.forward(x)

print(output.shape)  # Expected output: (2, 512, 768)
```
## References

[Medium](https://medium.com/@hhpatil001/transformers-from-scratch-in-simple-python-part-i-b290760c1040)  
[Pylessons](https://pylessons.com/transformers-introduction)  
[Machine Learning Mastery](https://machinelearningmastery.com/the-transformer-model/)
[Medium: LLM Foundations: Constructing and Training Decoder-Only Transformers](https://medium.com/@williamzebrowski7/llm-foundations-constructing-and-training-decoder-only-transformers-bfcc429b43a2)
[Medium: Building a GPT-Style Transformer Model from Scratch: My Deep Learning Journey](https://medium.com/@helloitsdaksh007/building-a-gpt-style-transformer-model-from-scratch-my-deep-learning-journey-91a98af9e5d0)
[Implementing a Decoder-Only Transformer from Scratch](https://github.com/shunzh/llm.ipynb/blob/main/llm.ipynb)
