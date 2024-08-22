from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoPeftModelForCausalLM.from_pretrained(
    "./../ft_models/llama_lora_netflix_item_v0/checkpoint-928",
    low_cpu_mem_usage=True,
)
# base_model = AutoModelForSequenceClassification.from_pretrained("./../ft_models/llama_lora_netflix_v0/merged_model", num_labels=1)
# model = PeftModel.from_pretrained(base_model, "./../ft_models/rlhf/netflix_final_rlhf_testmaskv1_step_1000")


tokenizer = AutoTokenizer.from_pretrained("./../ft_models/llama_lora_netflix_item_v0/checkpoint-928")

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("./../ft_models/llama_lora_netflix_item_v0/merged_model_928",safe_serialization=True)
tokenizer.save_pretrained("./../ft_models/llama_lora_netflix_item_v0/merged_model_928")
