import numpy as np
import transformers
import pathlib
import torch
import sys
import femr.models.transformer
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor
from .generate_labels import create_omop_meds_tutorial_arg_parser
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class CustomEarlyStoppingCallback(transformers.EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric) / state.best_metric
                > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1


def create_arg_parser():
    arg_parser = create_omop_meds_tutorial_arg_parser()
    arg_parser.add_argument(
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default=None
    )
    arg_parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=1e-5
    )
    arg_parser.add_argument(
        "--n_layers",
        dest="n_layers",
        type=int,
        default=11
    )
    arg_parser.add_argument(
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=50
    )
    arg_parser.add_argument(
        "--per_device_train_batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=1
    )
    arg_parser.add_argument(
        "--per_device_eval_batch_size",
        dest="per_device_eval_batch_size",
        type=int,
        default=1
    )
    arg_parser.add_argument(
        "--loss",
        action="store_true",
        help="If set, compute and save loss distributions for train and val batches instead of training."
    )
    arg_parser.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        default=None
    )
    arg_parser.add_argument(
        "--loss-level",
        dest="loss_level",
        choices=["batch", "sample"],
        default="batch",
        help="Compute loss at the 'batch' or 'sample' level."
    )
    return arg_parser


def main():
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)

    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    if args.model_path:
        tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(args.model_path, ontology=ontology)
    else:
        tokenizer_path = pretraining_data / 'tokenizer'
        tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
            tokenizer_path, ontology=ontology
        )

    task_path = pretraining_data / 'motor_task.pkl'
    with open(task_path, 'rb') as f:
        motor_task = pickle.load(f)

    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))

    # Finally, given the batches, we can train CLMBR.
    # We can use huggingface's trainer to do this.
    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer),
        n_layers=args.n_layers,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='swiglu',
    )

    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(
        transformer_config,
        motor_task.get_task_config()
    )

    if args.model_path:
        model = femr.models.transformer.FEMRModel.from_pretrained(args.model_path, task_config=motor_task.get_task_config())
    else:
        model = femr.models.transformer.FEMRModel(config)

    model = model.to(torch.device("cuda"))

    learning_rate = args.learning_rate
    output_dir = 'tmp_trainer_' + sys.argv[1]
    trainer_config = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        learning_rate=learning_rate,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=True,

        weight_decay=0.1,
        adam_beta2=0.95,

        report_to=["tensorboard"],

        num_train_epochs=args.n_epochs,

        warmup_steps=500,

        logging_strategy='epoch',
        logging_steps=10,

        save_strategy='epoch',
        eval_strategy='epoch',

        # prediction_loss_only=True,
        dataloader_num_workers=12,

        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_config,
        callbacks=[CustomEarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.001)],
    )

    if args.loss:
        model.eval()
        random.seed(42)
        k = 100 #5000

        # Randomly select k indices from train and val
        train_indices = random.sample(range(len(train_batches)), min(k, len(train_batches)))
        val_indices = random.sample(range(len(val_batches)), min(k, len(val_batches)))
        subset_train_batches = train_batches.select(train_indices)
        subset_val_batches = val_batches.select(val_indices)
        
        # subset_train_batches = train_batches
        # subset_val_batches = val_batches

        train_loss_file = f'train_loss_{args.loss_level}_{k}.npy'
        val_loss_file = f'val_loss_{args.loss_level}_{k}.npy'
        train_representations_file = f'train_representations_{args.loss_level}_{k}.npy'
        val_representations_file = f'val_representations_{args.loss_level}_{k}.npy'

        def to_device(data, device):
            if isinstance(data, dict):
                return {k: to_device(v, device) for k, v in data.items()}
            elif isinstance(data, torch.Tensor):
                return data.to(device)
            else:
                return data

        def compute_per_batch_losses(dataset, processor, model):
            losses = []
            representations = []
            for i in tqdm(range(len(dataset))):
                batch = processor.collate([dataset[i]])
                batch = to_device(batch["batch"], model.device)
                with torch.no_grad():
                    loss, result = model(batch, return_reprs=True)
                    if isinstance(loss, torch.Tensor):
                        losses.append(loss.item())
                    else:
                        losses.append(float(loss))
                    representations.append(result.get("representations", None).detach().cpu().numpy())
            return np.array(losses), np.array(representations)

        def compute_per_sample_losses(dataset, processor, model):
            losses = []
            representations = []
            for i in tqdm(range(len(dataset))):
                batch = processor.collate([dataset[i]])
                batch = to_device(batch["batch"], model.device)
                with torch.no_grad():
                    loss, result = model(batch, return_reprs=True)
                    per_sample_loss = result.get("per_sample_loss", None)
                    if per_sample_loss is not None:
                        losses.extend(per_sample_loss.detach().cpu().numpy())
                        representations.extend(result.get("representations", None).detach().cpu().numpy())
                    else:
                        raise RuntimeError("Per-sample loss is not available")
                        # fallback: use mean loss for all samples in batch
                        n_samples = batch["subject_ids"].shape[0]
                        losses.extend([loss.item()] * n_samples)
            return np.array(losses), np.array(representations)
        # def compute_per_sample_losses(dataset, processor, model):
        #     losses = []
        #     for i in tqdm(range(len(dataset))):
        #         batch_dict = dataset[i]
        #         n_samples = len(batch_dict['subject_ids'])
        #         for j in tqdm(range(n_samples)):
        #             single_sample = {k: v[j:j+1] if isinstance(v, np.ndarray) and v.shape[0] == n_samples else v
        #                              for k, v in batch_dict.items()}
        #             batch = processor.collate([single_sample])
        #             batch = to_device(batch["batch"], model.device)
        #             with torch.no_grad():
        #                 loss, _ = model(batch)
        #                 if isinstance(loss, torch.Tensor):
        #                     losses.append(loss.item())
        #                 else:
        #                     losses.append(float(loss))
        #     return np.array(losses)

        if args.loss_level == "batch":
            compute_loss_fn = compute_per_batch_losses
            loss_label = "batch"
        else:
            compute_loss_fn = compute_per_sample_losses
            loss_label = "sample"

        if not os.path.exists(train_loss_file) or not os.path.exists(train_representations_file):
            print(f"Computing per-{loss_label} loss for train set (random k)...")
            train_losses, train_representations = compute_loss_fn(subset_train_batches, processor, model)
            np.save(train_loss_file, train_losses)
            np.save(train_representations_file, train_representations)
            print(f"Saved train losses to {train_loss_file}")
        else:
            print(f"Loading existing train losses from {train_loss_file}")
            train_losses = np.load(train_loss_file)
            train_representations = np.load(train_representations_file)
        if not os.path.exists(val_loss_file) or not os.path.exists(val_representations_file):
            print(f"Computing per-{loss_label} loss for val set (random k)...")
            val_losses, val_representations = compute_loss_fn(subset_val_batches, processor, model)
            np.save(val_loss_file, val_losses)
            np.save(val_representations_file, val_representations)
            print(f"Saved val losses to {val_loss_file}")
        else:
            print(f"Loading existing val losses from {val_loss_file}")
            val_losses = np.load(val_loss_file)
            val_representations = np.load(val_representations_file)
            print(val_losses)

        if args.loss_level == "sample":
            train_losses = train_losses.mean(axis=(1,2))
            val_losses = val_losses.mean(axis=(1,2))
        print(train_losses.shape)
        print(val_losses.shape)
        print(train_representations.shape)
        print(val_representations.shape)

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(train_losses, bins=30, alpha=0.5, label='Train', color='blue')
        plt.hist(val_losses, bins=30, alpha=0.5, label='Validation', color='orange')
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.title(f'Loss Distribution: Train vs Validation (Random {k})')
        plt.legend()
        plt.tight_layout()
        fig_name = f'loss_distribution_comparison_{args.loss_level}_{k}.png'
        plt.savefig(fig_name)
        print(f"Saved loss distribution plot to {fig_name}")
        plt.close()
    else:
        train_result = trainer.train(resume_from_checkpoint=args.checkpoint_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
