"""Training pipeline — SFT / DPO / GRPO / PPO via transformers + TRL.

Launched by the repository when a training run is queued.
Updates run status in the artifact database as it progresses.
"""
import json, os, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
from artifact_repo.database import ArtifactDatabase


def _run_dpo(model, tokenizer, training_args, dataset, config,
             method, output_dir, db, run_id):
    """Run DPO (Direct Preference Optimization) training via TRL.

    DPO trains on preference pairs (chosen vs rejected responses)
    without needing a separate reward model. This is the simplest
    RL method to implement — reuses SFT infrastructure + preference data.

    Returns train_result on success, or None on failure (status already updated).
    """
    import torch
    from datasets import Dataset as HFDataset

    print(f"  [DPO] Preparing preference dataset...")

    # ── Build or load DPO-formatted dataset ──────────────────────────────
    # DPO needs: prompt, chosen, rejected columns
    # If the loaded dataset already has these columns, use it directly.
    # Otherwise, synthesize preference pairs from the text dataset.
    if hasattr(dataset, 'column_names') and 'chosen' in dataset.column_names:
        dpo_dataset = dataset
        print(f"  [DPO] Using existing preference dataset ({len(dpo_dataset)} pairs)")
    elif hasattr(dataset, 'column_names') and 'text' in dataset.column_names:
        # Synthesize chosen/rejected from single text column
        # For demo: split each text into prompt+chosen, generate a rejected variant
        rows = []
        for ex in dataset:
            text = ex['text']
            # Simple heuristic: last assistant turn is "chosen",
            # a deliberately worse variant is "rejected"
            parts = text.rsplit('Assistant:', 1) if 'Assistant:' in text else text.rsplit('\n', 1)
            prompt = parts[0].strip() if len(parts) > 1 else text[:len(text)//2]
            chosen = parts[1].strip() if len(parts) > 1 else text[len(text)//2:]
            # Generate a rejected variant with errors or inappropriate content
            rejected = chosen[:max(1, len(chosen)//2)] + " [TRUNCATED]"
            rows.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
        dpo_dataset = HFDataset.from_list(rows)
        print(f"  [DPO] Synthesized {len(dpo_dataset)} preference pairs from text")
    else:
        # Fallback: minimal demo dataset
        dpo_dataset = HFDataset.from_list([
            {'prompt': 'Explain quantum computing',
             'chosen': 'Quantum computing uses qubits that can exist in superposition states, enabling parallel computation across multiple states simultaneously.',
             'rejected': 'quantum is like magic computers idk'},
            {'prompt': 'What is the capital of France?',
             'chosen': 'The capital of France is Paris.',
             'rejected': 'paris lol'},
            {'prompt': 'Write a haiku about AI',
             'chosen': 'Silicon dreams wake / Neural pathways softly glow / Machine learns to think',
             'rejected': 'ai is cool / it does stuff / i like it'},
        ])
        print(f"  [DPO] Using fallback demo dataset ({len(dpo_dataset)} pairs)")

    # ── Tokenize for DPO format ──────────────────────────────────────────
    def tokenize_dpo(ex):
        """Tokenize prompt+chosen and prompt+rejected as separate sequences."""
        prompt_tok = tokenizer(ex['prompt'], truncation=True, max_length=256, padding=False)
        chosen_tok = tokenizer(ex['prompt'] + ' ' + ex['chosen'], truncation=True, max_length=512, padding=False)
        rejected_tok = tokenizer(ex['prompt'] + ' ' + ex['rejected'], truncation=True, max_length=512, padding=False)
        return {
            'prompt_input_ids': prompt_tok['input_ids'],
            'prompt_attention_mask': prompt_tok['attention_mask'],
            'chosen_input_ids': chosen_tok['input_ids'],
            'chosen_attention_mask': chosen_tok['attention_mask'],
            'rejected_input_ids': rejected_tok['input_ids'],
            'rejected_attention_mask': rejected_tok['attention_mask'],
        }

    dpo_dataset = dpo_dataset.map(tokenize_dpo, remove_columns=dpo_dataset.column_names)

    # ── Create DPOTrainer ────────────────────────────────────────────────
    try:
        import trl
        from trl import DPOTrainer, DPOConfig

        dpo_config = DPOConfig(
            output_dir=str(output_dir),
            learning_rate=config.get('learning_rate', 5e-6),
            per_device_train_batch_size=config.get('batch_size', 2),
            num_train_epochs=config.get('num_train_epochs', 2),
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to='none',
            beta=config.get('dpo_beta', 0.1),
            max_length=512,
            max_prompt_length=256,
        )

        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dpo_dataset,
            tokenizer=tokenizer,
        )
        print(f"  [DPO] DPOTrainer created (beta={dpo_config.beta})")

    except ImportError:
        db.update_training_run(run_id, status='failed',
                               metrics=json.dumps({'error': 'TRL DPOTrainer not available — install trl>=0.7.0'}))
        print("  [DPO] FAILED: TRL DPOTrainer not available")
        return None
    except Exception as e:
        db.update_training_run(run_id, status='failed',
                               metrics=json.dumps({'error': f'DPO setup failed: {e}'}))
        print(f"  [DPO] FAILED: {e}")
        return None

    return trainer


def train(run_id: int):
    db = ArtifactDatabase()
    run = db._conn.execute("SELECT * FROM training_runs WHERE id=?", (run_id,)).fetchone()
    if not run:
        print(f"Training run #{run_id} not found")
        return
    run = dict(run)
    base_model = db.get_model(run['base_model_id']) if run['base_model_id'] else None
    config = json.loads(run['config'] or '{}')

    print(f"Training run #{run_id}: {run['method']}")
    print(f"  Base model: {base_model['filename'] if base_model else 'N/A'}")
    print(f"  Dataset:    {run['dataset'] or 'N/A'}")
    print(f"  Config:     {config}")

    db.update_training_run(run_id, status='running')

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        import trl

        model_name = base_model['path'] if base_model else config.get('model_name', 'distilgpt2')
        dataset_name = run['dataset'] or config.get('dataset', '')
        method = run['method']
        output_dir = Path.home() / '.config' / 'unified-llm' / 'training' / f'run_{run_id}'
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map='auto' if torch.cuda.is_available() else 'cpu',
        )

        lora_config = LoraConfig(
            r=config.get('lora_rank', 16),
            lora_alpha=config.get('lora_alpha', 32),
            target_modules=config.get('target_modules', ['q_proj', 'v_proj']),
            lora_dropout=config.get('lora_dropout', 0.05),
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=config.get('learning_rate', 2e-5),
            per_device_train_batch_size=config.get('batch_size', 4),
            num_train_epochs=config.get('num_train_epochs', 3),
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to='none',
        )

        # Simple SFT trainer
        if dataset_name:
            dataset = load_dataset('text', data_files=dataset_name, split='train') if os.path.exists(dataset_name) else load_dataset(dataset_name, split='train')
        else:
            from datasets import Dataset
            dataset = Dataset.from_list([
                {'text': f"User: hello\nAssistant: Hi! How can I help?"},
                {'text': f"User: what is 2+2\nAssistant: 4"},
                {'text': f"User: write a poem\nAssistant: Roses are red..."},
            ])

        def tokenize_fn(ex):
            return tokenizer(ex['text'], truncation=True, padding='max_length', max_length=512)

        tokenized = dataset.map(tokenize_fn)

        if method == 'sft':
            trainer_cls = trl.SFTTrainer if hasattr(trl, 'SFTTrainer') else None
            if trainer_cls:
                trainer = trainer_cls(
                    model=model, args=training_args,
                    train_dataset=tokenized,
                    tokenizer=tokenizer,
                )
            else:
                from transformers import Trainer
                trainer = Trainer(
                    model=model, args=training_args,
                    train_dataset=tokenized,
                    tokenizer=tokenizer,
                )
        elif method == 'dpo':
            # ── DPO Training via TRL ──────────────────────────────────────
            trainer = _run_dpo(model, tokenizer, training_args, dataset, config,
                              method, output_dir, db, run_id)
            if trainer is None:
                return  # Failed — status already updated
        elif method in ('grpo', 'ppo'):
            db.update_training_run(run_id, status='failed',
                                   metrics=json.dumps({'error': f'{method} not yet implemented'}))
            return
        else:
            db.update_training_run(run_id, status='failed',
                                   metrics=json.dumps({'error': f'Unknown method {method}'}))
            return

        train_result = trainer.train()
        model.save_pretrained(str(output_dir / 'adapter'))
        tokenizer.save_pretrained(str(output_dir / 'adapter'))

        metrics = {
            'train_loss': float(train_result.training_loss) if hasattr(train_result, 'training_loss') else 0,
            'runtime': train_result.metrics.get('train_runtime', 0) if hasattr(train_result, 'metrics') else 0,
            'samples': len(dataset),
        }

        # Register adapter in database
        try:
            from artifact_repo.repository import ArtifactRepository
            repo = ArtifactRepository(db)
            repo.add_adapter(
                name=f"{method}_run_{run_id}",
                path=str(output_dir / 'adapter'),
                base_model_id=run['base_model_id'],
                method=method,
                rank=config.get('lora_rank', 16),
                config=config,
            )
        except Exception:
            pass

        db.update_training_run(run_id, status='completed',
                               metrics=json.dumps(metrics, default=str))
        print(f"Training complete: {json.dumps(metrics, indent=2)}")

    except ImportError as e:
        db.update_training_run(run_id, status='failed',
                               metrics=json.dumps({'error': f'Missing package: {e}'}))
        print(f"FAILED: {e}")
    except Exception as e:
        db.update_training_run(run_id, status='failed',
                               metrics=json.dumps({'error': str(e)}))
        print(f"FAILED: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(prog='artifact-trainer')
    parser.add_argument('--run', type=int, required=True, help='Training run ID')
    args = parser.parse_args()
    train(args.run)


if __name__ == '__main__':
    main()
