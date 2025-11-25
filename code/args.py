import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")

    # Data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--context_max_length", type=int, help="Max context length in dataset.")
    parser.add_argument("--max_length", type=int, help="Max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, help="Max prompt length.")
    parser.add_argument("--entity_max_length", type=int, help="Max entity length in dataset.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer to use.")
    parser.add_argument("--text_tokenizer", type=str, help="Text tokenizer to use.")

    # Model
    parser.add_argument("--model", type=str, required=False, help="Path to pretrained model or model identifier.")
    parser.add_argument("--text_encoder", type=str, help="Text encoder to use.")
    parser.add_argument("--prompt_encoder", type=str, help="Prompt encoder to use.")

    # Optimization
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. Overrides num_train_epochs if set.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of update steps to accumulate before a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--num_warmup_steps", type=int, default=10000, help="Number of warmup steps.")
    parser.add_argument("--fp16", action='store_true', help="Use 16-bit (mixed) precision training.")

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases.")
    parser.add_argument("--entity", type=str, help="Weights & Biases entity (username or team name).")
    parser.add_argument("--project", type=str, help="Weights & Biases project name.")
    parser.add_argument("--name", type=str, help="Weights & Biases run name.")
    

    args = parser.parse_args()
    return args