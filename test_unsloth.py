import torch
import gc

print("=" * 60)
print("GPU & MODEL LOADING TEST")
print("=" * 60)

# ============ STEP 1: Basic GPU Check ============
print("\n[1/4] Checking GPU availability...")
if not torch.cuda.is_available():
    print("❌ CUDA not available! Check your PyTorch installation.")
    exit(1)

print(f"✅ GPU Found: {torch.cuda.get_device_name(0)}")
print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"   VRAM Free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

# ============ STEP 2: Test Unsloth Import ============
print("\n[2/4] Testing Unsloth import...")
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth imported successfully")
except Exception as e:
    print(f"❌ Failed to import Unsloth: {e}")
    exit(1)

# ============ STEP 3: Load Model in 4-bit ============
print("\n[3/4] Loading LLaMA 3.2 1B in 4-bit mode...")
print("   (This will download ~1.2GB on first run)")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    print("✅ Model loaded successfully")
    
    # Check VRAM usage after loading
    vram_used = (torch.cuda.memory_allocated(0) / 1024**3)
    vram_reserved = (torch.cuda.memory_reserved(0) / 1024**3)
    print(f"   VRAM Used: {vram_used:.2f} GB")
    print(f"   VRAM Reserved: {vram_reserved:.2f} GB")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# ============ STEP 4: Add LoRA Adapters ============
print("\n[4/4] Adding LoRA adapters...")

try:
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print("✅ LoRA adapters added successfully")
    
    # Final VRAM check
    vram_used = (torch.cuda.memory_allocated(0) / 1024**3)
    vram_reserved = (torch.cuda.memory_reserved(0) / 1024**3)
    vram_free = (torch.cuda.mem_get_info()[0] / 1024**3)
    
    print(f"   VRAM Used: {vram_used:.2f} GB")
    print(f"   VRAM Reserved: {vram_reserved:.2f} GB")
    print(f"   VRAM Free: {vram_free:.2f} GB")
    
except Exception as e:
    print(f"❌ Failed to add LoRA: {e}")
    exit(1)

# ============ BONUS: Quick Inference Test ============
print("\n[BONUS] Testing inference...")

try:
    # Simple test prompt
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("✅ Inference works!")
    print(f"   Test output: {response[-100:]}")  # Last 100 chars
    
except Exception as e:
    print(f"⚠️  Inference test failed: {e}")
    print("   (This is non-critical - training may still work)")

# ============ FINAL SUMMARY ============
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("✅ GPU: Working")
print("✅ Unsloth: Working")
print("✅ Model Loading: Working")
print("✅ LoRA: Working")
print(f"✅ VRAM Available for Training: ~{vram_free:.1f} GB")

if vram_free < 2.0:
    print("\n⚠️  WARNING: Less than 2GB VRAM free")
    print("   Training may run out of memory")
    print("   Consider reducing batch_size to 1")
else:
    print("\n🎉 All checks passed! Ready to train.")
    print("   You can proceed with fine-tuning.")

print("=" * 60)

# Clean up
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()