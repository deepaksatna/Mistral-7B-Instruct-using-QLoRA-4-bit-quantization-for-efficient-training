#!/usr/bin/env python3
"""
Distributed Training Benchmark for Mistral-7B QLoRA
Tests multi-node training performance across 2 and 4 A10 GPU nodes
"""

import os
import sys
import time
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import argparse
from datetime import datetime
import psutil
import pynvml


class DistributedBenchmark:
    """Benchmark distributed training across multiple nodes"""

    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.setup_device()
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "node_id": os.environ.get("NODE_ID", "unknown"),
        }

    def setup_distributed(self):
        """Initialize distributed process group"""
        # Get distributed parameters from environment
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Initialize process group
        if self.world_size > 1:
            dist.init_process_group(backend="nccl")

        if self.rank == 0:
            print(f"Distributed setup:")
            print(f"  World size: {self.world_size}")
            print(f"  Rank: {self.rank}")
            print(f"  Local rank: {self.local_rank}")

    def setup_device(self):
        """Setup GPU device"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        if self.rank == 0:
            print(f"Using device: {self.device}")

    def log_metric(self, key, value):
        """Log metric"""
        self.metrics[key] = value
        if self.rank == 0:
            print(f"  {key}: {value}")

    def get_gpu_memory(self):
        """Get GPU memory usage"""
        if not torch.cuda.is_available():
            return {}

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.local_rank) / 1e9
        reserved = torch.cuda.memory_reserved(self.local_rank) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.local_rank) / 1e9

        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_allocated_gb": round(max_allocated, 2)
        }

    def create_synthetic_dataset(self, num_samples=1000):
        """Create synthetic dataset for benchmarking"""
        if self.rank == 0:
            print(f"\nCreating synthetic dataset ({num_samples} samples)...")

        prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "What are the benefits of exercise?",
            "Describe the water cycle.",
            "How does machine learning work?"
        ]

        data = {
            "text": [prompts[i % len(prompts)] for i in range(num_samples)]
        }

        return Dataset.from_dict(data)

    def load_model_and_tokenizer(self):
        """Load Mistral-7B with QLoRA configuration"""
        if self.rank == 0:
            print("\nLoading model and tokenizer...")

        model_path = self.args.model_path

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Load model
        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": self.local_rank},
            torch_dtype=torch.bfloat16
        )
        load_time = time.time() - load_start

        if self.rank == 0:
            print(f"Model loaded in {load_time:.2f}s")

        self.log_metric("model_load_time_sec", round(load_time, 2))
        self.log_metric("model_memory_after_load", self.get_gpu_memory())

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )

        # Add LoRA adapters
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)

        if self.rank == 0:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return model, tokenizer

    def tokenize_dataset(self, dataset, tokenizer):
        """Tokenize dataset"""
        if self.rank == 0:
            print(f"\nTokenizing dataset (seq_length={self.args.seq_length})...")

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.args.seq_length,
                padding="max_length",
                return_tensors="pt"
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized

    def run_benchmark(self):
        """Run distributed training benchmark"""
        if self.rank == 0:
            print("=" * 80)
            print("DISTRIBUTED TRAINING BENCHMARK")
            print("=" * 80)
            print(f"Configuration:")
            print(f"  World size: {self.world_size} nodes")
            print(f"  Batch size per device: {self.args.batch_size}")
            print(f"  Global batch size: {self.args.batch_size * self.world_size}")
            print(f"  Sequence length: {self.args.seq_length}")
            print(f"  LoRA rank: {self.args.lora_rank}")
            print(f"  Training steps: {self.args.max_steps}")
            print("=" * 80)

        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Create dataset
        dataset = self.create_synthetic_dataset(num_samples=self.args.num_samples)
        tokenized_dataset = self.tokenize_dataset(dataset, tokenizer)

        # Training arguments
        output_dir = f"/tmp/distributed_benchmark_rank{self.rank}"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            max_steps=self.args.max_steps,
            learning_rate=2e-4,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=10,
            save_steps=1000,
            save_total_limit=1,
            warmup_steps=10,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,  # Important for LoRA
            report_to="none",
            disable_tqdm=(self.rank != 0)  # Only show progress on rank 0
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )

        # Synchronize before training
        if self.world_size > 1:
            dist.barrier()

        # Training
        if self.rank == 0:
            print("\nStarting training...")

        train_start = time.time()

        try:
            train_result = trainer.train()
            train_time = time.time() - train_start

            # Calculate metrics
            samples_trained = self.args.max_steps * self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps
            samples_per_sec = samples_trained / train_time
            tokens_per_sec = samples_per_sec * self.args.seq_length

            # Log metrics
            self.log_metric("train_time_sec", round(train_time, 2))
            self.log_metric("samples_trained", samples_trained)
            self.log_metric("samples_per_second", round(samples_per_sec, 2))
            self.log_metric("tokens_per_second", round(tokens_per_sec, 2))
            self.log_metric("train_loss", round(train_result.training_loss, 4))
            self.log_metric("memory_peak", self.get_gpu_memory())

            # Calculate scaling efficiency (if world_size > 1)
            if self.world_size > 1:
                # Compare to expected linear scaling
                expected_samples_per_sec = 1.46 * self.world_size  # Baseline from single node
                efficiency = (samples_per_sec / expected_samples_per_sec) * 100
                self.log_metric("scaling_efficiency_percent", round(efficiency, 2))

            # Cost calculation (OCI A10 pricing: $0.60/hour)
            cost_per_hour = 0.60
            cost_total = (train_time / 3600) * cost_per_hour * self.world_size
            cost_per_1k_samples = (cost_total / samples_trained) * 1000

            self.log_metric("cost_total_usd", round(cost_total, 4))
            self.log_metric("cost_per_1k_samples_usd", round(cost_per_1k_samples, 4))

            if self.rank == 0:
                print("\n" + "=" * 80)
                print("BENCHMARK COMPLETE")
                print("=" * 80)
                print(f"Training time: {train_time:.2f}s")
                print(f"Samples/sec: {samples_per_sec:.2f}")
                print(f"Tokens/sec: {tokens_per_sec:.0f}")
                if self.world_size > 1:
                    print(f"Scaling efficiency: {efficiency:.2f}%")
                print(f"Cost: ${cost_total:.4f} (${cost_per_1k_samples:.4f} per 1K samples)")
                print("=" * 80)

        except Exception as e:
            if self.rank == 0:
                print(f"\nError during training: {e}")
            self.log_metric("error", str(e))

        # Synchronize before cleanup
        if self.world_size > 1:
            dist.barrier()

    def save_results(self):
        """Save benchmark results to JSON"""
        self.metrics["end_time"] = datetime.now().isoformat()

        # Save results for this rank
        results_dir = "/results/distributed"
        os.makedirs(results_dir, exist_ok=True)

        output_file = f"{results_dir}/node{self.rank}_world{self.world_size}_batch{self.args.batch_size}_seq{self.args.seq_length}.json"

        with open(output_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        if self.rank == 0:
            print(f"\nResults saved to: {output_file}")

    def cleanup(self):
        """Cleanup distributed resources"""
        if self.world_size > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed Training Benchmark")

    # Model config
    parser.add_argument("--model-path", type=str,
                       default="/models/mistralai--Mistral-7B-Instruct-v0.3",
                       help="Path to pre-loaded model")

    # Training config
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size per device")
    parser.add_argument("--seq-length", type=int, default=512,
                       help="Sequence length")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Number of training steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Gradient accumulation steps")

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")

    # Dataset config
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of synthetic samples")

    args = parser.parse_args()

    # Run benchmark
    benchmark = DistributedBenchmark(args)

    try:
        benchmark.run_benchmark()
        benchmark.save_results()
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
