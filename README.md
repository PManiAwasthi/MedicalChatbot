# MedicalChatbot

This project utilizes OpenSource Libraries, Models, and APIs to create and run a Chatbot specifically designed for medical assistance. Using LangChain, HuggingFace, Llama model, Pinecone, etc.

To create a new conda environment
```
conda create -p env python=3.8 -y
```

to activate a conda environment
```
conda activate ./env
```

installing necessary packages
```
pip install -r requirements.txt
```

if you have not downloaded the quantized version of the model
```
from huggingface_hub import hf_hub_download

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
```