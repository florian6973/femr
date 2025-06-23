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
from torch.nn.utils._per_sample_grad import call_for_per_sample_grads
from torch.func import functional_call, vmap, grad
from opacus.grad_sample import GradSampleModule

from femr.omop_meds_tutorial.generate_labels import create_omop_meds_tutorial_arg_parser
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
    arg_parser.add_argument(
        "--gradient-attack",
        action="store_true",
        help="Compute and save gradient for attack."
    )
    arg_parser.add_argument(
        "--last-layer-gradients",
        action="store_true",
        help="If set, only keep gradients from the last layer instead of all parameters."
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

    # Display first element of val_batches
    print("=" * 50)
    print("FIRST ELEMENT OF VAL_BATCHES")
    print("=" * 50)
    print(val_batches[0])
    print(val_batches[0]['transformer']['timestamps'].shape)
    print("\nKeys in the first element:")
    for key in val_batches[0].keys():
        print(f"  - {key}")

    # Finally, given the batches, we can train CLMBR.
    # We can use huggingface's trainer to do this.
    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer),
        # n_layers=args.n_layers,
        n_layers=16,
        n_heads=32,
        hidden_size=3072,
        intermediate_size=12288,
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

    # Print model architecture and parameter count
    print("=" * 50)
    print("MODEL ARCHITECTURE")
    print("=" * 50)
    print(model)
    
    print("\n" + "=" * 50)
    print("PARAMETER COUNT")
    print("=" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print parameter count by module
    print("\n" + "=" * 50)
    print("PARAMETERS BY MODULE")
    print("=" * 50)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name}: {params:,} parameters")
    
    print("=" * 50)

    model = model.to(torch.device("cuda"))

    # Wrap with Opacus GradSampleModule if needed
    # if args.gradient_attack and args.last_layer_gradients:
        # model = GradSampleModule(model)

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
        # k = 100 #5000
        k = 2 #5000

        # print(train_batches)
        # print(val_batches)
        # input()

        # Randomly select k indices from train and val
        train_indices = random.sample(range(len(train_batches)), min(k, len(train_batches)))
        val_indices = random.sample(range(len(val_batches)), min(k, len(val_batches)))
        subset_train_batches = train_batches.select(train_indices)
        subset_val_batches = val_batches.select(val_indices)
        
        # subset_train_batches = train_batches
        # subset_val_batches = val_batches

        train_loss_file = f'train_loss_{args.loss_level}_{k}.npy'
        val_loss_file = f'val_loss_{args.loss_level}_{k}.npy'
        train_gradients_file = f'train_gradients_{args.loss_level}_{k}.npy'
        val_gradients_file = f'val_gradients_{args.loss_level}_{k}.npy'
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
            gradients = []
            for i in tqdm(range(len(dataset))):
                batch = processor.collate([dataset[i]])
                batch = to_device(batch["batch"], model.device)
                if not args.gradient_attack:
                    with torch.no_grad():
                        loss, result = model(batch, return_reprs=True)
                        if isinstance(loss, torch.Tensor):
                            losses.append(loss.item())
                            # compute average loss per batch
                            # losses.append(loss.item() / batch["subject_ids"].shape[1])
                        else:
                            losses.append(float(loss))
                        representations.append(result.get("representations", None).detach().cpu().numpy())
                else:
                    loss, result = model(batch, return_reprs=True)
                    grads = result.get("gradients", None)
                    if grads is not None:
                        gradients.extend(grads.detach().cpu().numpy())
                    else:
                        # Compute gradients manually
                        model.zero_grad()
                        loss.backward()
                        grad_list = []
                        if args.last_layer_gradients:
                            # Only keep gradients from the last layer
                            last_param = None
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    last_param = param
                            if last_param is not None:
                                grad_list.append(last_param.grad.detach().cpu().numpy().flatten())
                        else:
                            # Keep all gradients
                            for param in model.parameters():
                                if param.grad is not None:
                                    grad_list.append(param.grad.detach().cpu().numpy().flatten())
                        if grad_list:
                            gradients.append(np.concatenate(grad_list))
                        else:
                            print("Warning: No gradients found for batch", i)
                    # losses.extend(result.get("per_sample_loss", None).detach().cpu().numpy())
                    losses.append(loss.item())
                    representations.extend(result.get("representations", None).detach().cpu().numpy())
            return np.array(losses), np.array(representations), np.array(gradients)

        def compute_per_sample_losses(dataset, processor, model):
            losses = []
            representations = []
            gradients = []

            for i in tqdm(range(len(dataset))):
                batch = processor.collate([dataset[i]])
                batch = to_device(batch["batch"], model.device)
                if not args.gradient_attack:
                    with torch.no_grad():
                        loss, result = model(batch, return_reprs=True, return_per_sample_loss=True)
                        per_sample_loss = result.get("per_sample_loss", None)
                        if per_sample_loss is not None:
                            losses.extend(per_sample_loss.detach().cpu().numpy())
                            representations.extend(result.get("representations", None).detach().cpu().numpy())
                        else:
                            raise RuntimeError("Per-sample loss is not available")
                            # fallback: use mean loss for all samples in batch
                            n_samples = batch["subject_ids"].shape[1]
                            losses.extend([loss.item()] * n_samples)
                else:
                    # print(batch["subject_ids"].shape)
                    # with torch.no_grad():
                    #     loss, result = model(batch, return_reprs=True)       
                    # losses.extend(result.get("per_sample_loss", None).detach().cpu().numpy())
                    # representations.extend(result.get("representations", None).detach().cpu().numpy())
                    # print(np.array(losses).shape)
                    # print("end")
                    # exit()
                    if args.last_layer_gradients:
                        ## TODO: Implement this
                        # model.zero_grad()
                        # loss.backward()
                        # # Find last parameter with grad_sample
                        # last_param = None
                        # for name, param in model.named_parameters():
                        #     if hasattr(param, "grad_sample") and param.grad_sample is not None:
                        #         last_param = param
                        # if last_param is not None:
                        #     # grad_sample shape: [batch_size, ...]
                        #     gradients.append(last_param.grad_sample.detach().cpu().numpy())
                        # else:
                        #     print(f"No grad_sample found for last layer in batch {i}")

                        # model_grad.zero_grad()
                        # loss_grad, result_grad = model_grad(batch)                        
                        # loss_grad.backward()
                        # last_param = None
                        # for name, param in model.named_parameters():
                        #     if param.grad is not None:
                        #         last_param = param
                        # if last_param is not None:
                        #     gradients.append(last_param.grad_sample.detach().cpu().numpy().flatten())
                        # else:
                        #     raise RuntimeError("No gradients found for the last layer")

                        # Compute per-sample gradients for the last layerbatch_dict = dataset[i]
                        batch_dict = dataset[i]
                        n_samples = len(batch_dict['subject_ids'])
                        print(batch_dict['subject_ids'])
                        for j in tqdm(range(n_samples)):
                            # single_sample = {k: v[j:j+1] if isinstance(v, np.ndarray) and v.shape[0] == n_samples else v
                            #  for k, v in batch_dict.items()}
                            for batch_key, batch_value in batch_dict.items():
                                print(batch_key, batch_value.shape if isinstance(batch_value, torch.Tensor) else type(batch_value))
                                if isinstance(batch_value, dict):
                                    for batch_key, batch_value in batch_value.items():
                                        print(" ", batch_key, batch_value.shape if isinstance(batch_value, torch.Tensor) else type(batch_value))
                                        if isinstance(batch_value, dict):
                                            for batch_key, batch_value in batch_value.items():
                                                print("  ", batch_key, batch_value.shape if isinstance(batch_value, torch.Tensor) else type(batch_value))
                                                if isinstance(batch_value, dict):
                                                    for batch_key, batch_value in batch_value.items():
                                                        print("   ", batch_key, batch_value.shape if isinstance(batch_value, torch.Tensor) else type(batch_value))
                                                        
                            # print(batch_dict)
                            input()
                            # print(single_sample)
                            single_sample = batch_dict
                            input()
                            single_sample = processor.collate([single_sample])
                            single_sample = to_device(single_sample["batch"], model.device)
                            model.zero_grad()
                            single_loss, single_result = model(single_sample, return_reprs=True)
                            losses.append(single_loss.item())
                            representations.append(single_result.get("representations", None).detach().cpu().numpy())
                            single_loss.backward()
                            # Find last parameter with grad
                            last_param = None
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    last_param = param
                            if last_param is not None:
                                # print(f"Found gradients for sample {j} in batch {i}")
                                # print(last_param.grad.shape)
                                gradients.append(last_param.grad.detach().cpu().numpy().flatten())
                                print(gradients)
                                print(losses)
                                input()
                            else:
                                print(f"Warning: No gradients found for sample {j} in batch {i}")
                    else:
                        # Compute per-sample gradients for all parameters
                        # batch_size = batch["subject_ids"].shape[1]
                        batch_dict = dataset[i]
                        n_samples = len(batch_dict['subject_ids'])
                        for j in range(n_samples):
                            single_sample = {k: v[j:j+1] if isinstance(v, np.ndarray) and v.shape[0] == n_samples else v
                             for k, v in batch_dict.items()}
                            single_sample = processor.collate([single_sample])
                            single_sample = to_device(single_sample["batch"], model.device)
                            model.zero_grad()
                            single_loss, single_result = model(single_sample, return_reprs=True)
                            single_loss.backward()
                            grad_list = []
                            for param in model.parameters():
                                if param.grad is not None:
                                    grad_list.append(param.grad.detach().cpu().numpy().flatten())
                            if grad_list:
                                gradients.append(np.concatenate(grad_list))
                            else:
                                print(f"Warning: No gradients found for sample {j} in batch {i}")
           # return np.array(losses).mean(axis=(1,2)), np.array(representations), np.array(gradients)
            return np.array(losses).mean(axis=(1,2)), np.array(representations), np.array(gradients)
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


        if not os.path.exists(train_loss_file) or not os.path.exists(train_representations_file) or not os.path.exists(train_gradients_file):
            print(f"Computing per-{loss_label} loss for train set (random k)...")
            train_losses, train_representations, train_gradients = compute_loss_fn(subset_train_batches, processor, model)
            np.save(train_loss_file, train_losses)
            np.save(train_representations_file, train_representations)
            np.save(train_gradients_file, train_gradients)
            print(f"Saved train losses to {train_loss_file}")
        else:
            print(f"Loading existing train losses from {train_loss_file}")
            train_losses = np.load(train_loss_file)
            train_representations = np.load(train_representations_file)
            train_gradients = np.load(train_gradients_file)
        if not os.path.exists(val_loss_file) or not os.path.exists(val_representations_file) or not os.path.exists(val_gradients_file):
            print(f"Computing per-{loss_label} loss for val set (random k)...")
            val_losses, val_representations, val_gradients = compute_loss_fn(subset_val_batches, processor, model)
            np.save(val_loss_file, val_losses)
            np.save(val_representations_file, val_representations)
            np.save(val_gradients_file, val_gradients)
            print(f"Saved val losses to {val_loss_file}")
        else:
            print(f"Loading existing val losses from {val_loss_file}")
            val_losses = np.load(val_loss_file)
            val_representations = np.load(val_representations_file)
            val_gradients = np.load(val_gradients_file)
            print(val_losses)

        if train_losses.ndim == 3:
            train_losses = train_losses.mean(axis=(1,2))
            val_losses = val_losses.mean(axis=(1,2))
        print(train_losses.shape)
        print(val_losses.shape)
        print(train_representations.shape)
        print(val_representations.shape)
        print(train_gradients.shape)
        print(val_gradients.shape)
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
