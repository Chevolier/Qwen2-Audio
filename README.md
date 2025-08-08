
# Training
## Environment Preparation

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
conda create -n llama-factory -y
conda activate llama-factory
pip install -e ".[torch,metrics]" --no-build-isolation

```

## Training

```bash
export FORCE_TORCHRUN=1
llamafactory-cli train examples/train_full/qwen2_audio_full_sft.yaml
```

# Inference


