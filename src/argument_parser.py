import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument(
        "--model_card", type=str, default="google/mt5-small", help="Model to be used"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Mini Batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight Decay")
    parser.add_argument(
        "--learning_rate", type=float, default=5.6e-5, help="Learning rate"
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=3, help="Save total limit"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--output_dir", type=str, default=".", help="Output_directory")
    args = parser.parse_args()
    return args
