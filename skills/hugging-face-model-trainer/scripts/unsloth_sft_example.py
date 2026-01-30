# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl>=0.12.0",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
# ]
# ///
"""
Fine-tune Vision Language Models (VLMs) using Unsloth optimizations.

Unsloth provides ~2x faster training and ~60% less VRAM usage.
Supports streaming datasets - no disk space needed for large VLM datasets.

USAGE:

  Run locally (requires GPU):
    uv run unsloth_sft_example.py \\
        --max-steps 100 \\
        --output-repo username/vlm-test

  Run on HF Jobs:
    hf jobs uv run \\
        https://huggingface.co/path/to/unsloth_sft_example.py \\
        --flavor a10g-large --secrets HF_TOKEN \\
        -- --max-steps 500 --output-repo username/vlm-finetuned

  With Trackio monitoring:
    uv run unsloth_sft_example.py \\
        --max-steps 500 \\
        --output-repo username/vlm-finetuned \\
        --trackio-space username/trackio

SUPPORTED MODELS:
  - Vision: unsloth/gemma-3-4b-pt, unsloth/Qwen3-VL-8B-Instruct, etc.
  - Text: unsloth/Qwen2.5-7B-bnb-4bit, unsloth/Llama-3.2-3B-bnb-4bit, etc.

See: https://unsloth.ai/docs for full model list and documentation.
"""

import argparse
import logging
import os
import sys
import time

# Force unbuffered output for HF Jobs logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_cuda():
    """Check CUDA availability and exit if not available."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Run on HF Jobs with --flavor a10g-large or similar.")
        sys.exit(1)
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune VLMs with Unsloth optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (local)
  uv run unsloth_sft_example.py --max-steps 50 --output-repo username/test

  # Full training with monitoring
  uv run unsloth_sft_example.py \\
      --max-steps 500 \\
      --output-repo username/vlm-finetuned \\
      --trackio-space username/trackio

  # Custom model and dataset
  uv run unsloth_sft_example.py \\
      --base-model unsloth/Qwen3-VL-8B-Instruct \\
      --dataset your-username/your-vlm-dataset \\
      --max-steps 1000 \\
      --output-repo username/custom-vlm
        """,
    )

    # Model and data
    parser.add_argument(
        "--base-model",
        default="unsloth/gemma-3-4b-pt",
        help="Base VLM model. Use Unsloth variants for best performance. (default: unsloth/gemma-3-4b-pt)",
    )
    parser.add_argument(
        "--chat-template",
        default="gemma-3",
        help="Chat template to apply. Options: gemma-3, qwen3-vl, llama-3, etc. (default: gemma-3)",
    )
    parser.add_argument(
        "--dataset",
        default="davanstrien/iconclass-vlm-sft",
        help="VLM dataset with 'images' and 'messages' columns (default: davanstrien/iconclass-vlm-sft)",
    )
    parser.add_argument(
        "--output-repo",
        required=True,
        help="HF Hub repo to push model to (e.g., 'username/vlm-finetuned')",
    )

    # Training config
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Training steps (default: 500). Required for streaming datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4). Effective batch = batch-size * this",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    # LoRA config
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16). Higher = more capacity but more VRAM",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32). Usually 2*r",
    )

    # Logging and output
    parser.add_argument(
        "--trackio-space",
        default=None,
        help="HF Space for Trackio dashboard (e.g., 'username/trackio')",
    )
    parser.add_argument(
        "--save-local",
        default="unsloth-vlm-output",
        help="Local directory to save model (default: unsloth-vlm-output)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable dataset streaming (downloads full dataset)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Unsloth VLM Fine-tuning")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Base model:      {args.base_model}")
    print(f"  Chat template:   {args.chat_template}")
    print(f"  Dataset:         {args.dataset}")
    print(f"  Max steps:       {args.max_steps}")
    print(
        f"  Batch size:      {args.batch_size} x {args.gradient_accumulation} "
        f"= {args.batch_size * args.gradient_accumulation} effective"
    )
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  LoRA rank:       {args.lora_r}")
    print(f"  Output repo:     {args.output_repo}")
    print(f"  Trackio space:   {args.trackio_space or '(not configured)'}")
    print(f"  Streaming:       {not args.no_streaming}")
    print()

    # Check CUDA before heavy imports
    check_cuda()

    # Enable fast transfers
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Import heavy dependencies
    from unsloth import FastVisionModel, get_chat_template
    from unsloth.trainer import UnslothVisionDataCollator
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import login
    import trackio

    # Login to Hub
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face Hub")
    else:
        logger.warning("HF_TOKEN not set - model upload may fail")

    # Initialize Trackio if configured
    if args.trackio_space:
        trackio.init(
            project="unsloth-vlm-training",
            name=f"{args.base_model.split('/')[-1]}-{args.max_steps}steps",
            space_id=args.trackio_space,
            config={
                "model": args.base_model,
                "dataset": args.dataset,
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
            },
        )
        logger.info(f"Trackio dashboard: https://huggingface.co/spaces/{args.trackio_space}")

    # 1. Load model
    print("\n[1/5] Loading model with Unsloth optimizations...")
    start = time.time()

    model, processor = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # Add LoRA adapters for all modalities
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    # Apply chat template
    processor = get_chat_template(processor, args.chat_template)
    print(f"Model loaded in {time.time() - start:.1f}s")

    # 2. Load dataset
    print("\n[2/5] Loading dataset...")
    start = time.time()

    streaming = not args.no_streaming
    dataset = load_dataset(
        args.dataset,
        split="train",
        streaming=streaming,
    )

    if streaming:
        # Peek at first sample to show info
        sample = next(iter(dataset))
        print(f"Streaming dataset ready in {time.time() - start:.1f}s")
        if "messages" in sample:
            print(f"  Sample has {len(sample['messages'])} messages")
        if "images" in sample:
            img_count = len(sample["images"]) if isinstance(sample["images"], list) else 1
            print(f"  Sample has {img_count} image(s)")

        # Reload dataset (consumed one sample above)
        dataset = load_dataset(args.dataset, split="train", streaming=True)
    else:
        print(f"Dataset loaded in {time.time() - start:.1f}s")
        print(f"  {len(dataset)} examples")

    # 3. Configure trainer
    print("\n[3/5] Configuring trainer...")

    # Enable training mode
    FastVisionModel.for_training(model)

    training_config = SFTConfig(
        output_dir=args.save_local,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=max(1, args.max_steps // 20),
        save_strategy="steps",
        save_steps=max(100, args.max_steps // 5),
        optim="adamw_torch_fused",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        # VLM-specific settings (required for Unsloth)
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_seq_length,
        # Hub settings
        push_to_hub=True,
        hub_model_id=args.output_repo,
        hub_strategy="checkpoint",
        # Logging
        report_to="trackio" if args.trackio_space else "none",
        run_name=f"unsloth-vlm-{args.max_steps}steps",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=training_config,
    )

    # 4. Train
    print(f"\n[4/5] Training for {args.max_steps} steps...")
    print(f"  Logging every {training_config.logging_steps} steps")
    print(f"  Saving every {training_config.save_steps} steps")
    start = time.time()

    trainer.train()

    train_time = time.time() - start
    print(f"\nTraining completed in {train_time / 60:.1f} minutes")
    print(f"  Speed: {args.max_steps / train_time:.2f} steps/s")

    # 5. Save and push
    print("\n[5/5] Saving model...")

    # Save locally
    model.save_pretrained(args.save_local)
    processor.save_pretrained(args.save_local)
    print(f"Saved locally to {args.save_local}/")

    # Push to Hub
    print(f"\nPushing to {args.output_repo}...")
    model.push_to_hub(args.output_repo)
    processor.push_to_hub(args.output_repo)
    print(f"Model available at: https://huggingface.co/{args.output_repo}")

    # Finish Trackio
    if args.trackio_space:
        trackio.finish()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"\nModel: https://huggingface.co/{args.output_repo}")
    if args.trackio_space:
        print(f"Metrics: https://huggingface.co/spaces/{args.trackio_space}")


if __name__ == "__main__":
    # Show help if no arguments
    if len(sys.argv) == 1:
        print("=" * 70)
        print("Unsloth VLM Fine-tuning")
        print("=" * 70)
        print("\nFine-tune Vision-Language Models with Unsloth optimizations.")
        print("~2x faster training and ~60% less VRAM vs standard methods.")
        print("\nFeatures:")
        print("  - Streaming datasets (no disk space needed)")
        print("  - 4-bit quantization for memory efficiency")
        print("  - LoRA for all modalities (vision + language)")
        print("  - Trackio integration for monitoring")
        print("  - Automatic Hub push")
        print("\nQuick start:")
        print("\n  uv run unsloth_sft_example.py \\")
        print("      --max-steps 500 \\")
        print("      --output-repo your-username/vlm-finetuned")
        print("\nHF Jobs (cloud GPU):")
        print("\n  hf jobs uv run <script-url> \\")
        print("      --flavor a10g-large --secrets HF_TOKEN \\")
        print("      -- --max-steps 500 --output-repo your-username/vlm-finetuned")
        print("\nFor full help: uv run unsloth_sft_example.py --help")
        print("=" * 70)
        sys.exit(0)

    main()
