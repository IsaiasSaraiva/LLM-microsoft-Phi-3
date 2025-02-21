🤖 LLM - Microsoft Phi-3 Mini 4K Instruct 🚀



🔍 Sobre o Projeto
Este repositório fornece um ambiente otimizado para executar o Microsoft Phi-3 Mini 4K Instruct, um modelo leve e eficiente projetado para tarefas de inferência com desempenho otimizado. Com suporte a CUDA, bitsandbytes, accelerate e outras otimizações, você pode rodar este modelo com maior eficiência em sua GPU. 💡⚡

🚀 Características

✅ Modelo: Microsoft Phi-3 Mini 4K Instruct 🤖
✅ Frameworks: PyTorch + Transformers + Accelerate + bitsandbytes
✅ Otimizações: Suporte a quantização 4-bit para reduzir uso de memória
✅ Compatibilidade: GPUs NVIDIA com suporte a CUDA
✅ Inferência eficiente e rápida

🛠️ Configuração e Instalação

🔹 1. Clone o Repositório

 git clone https://github.com/IsaiasSaraiva/LLM-microsoft-Phi-3
 cd LLM-microsoft-Phi-3

🔹 2. Crie um Ambiente Virtual (Recomendado)

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

🔹 3. Instale as Dependências

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes

🔹 4. Baixe o Modelo

from transformers import AutoModelForCausalLM, AutoTokenizer

id_model = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(id_model, device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(id_model)

🔹 5. Rode um Exemplo

input_text = "Explique a teoria da evolução de forma simples."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_length=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))

