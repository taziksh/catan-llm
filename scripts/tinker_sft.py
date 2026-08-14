"""Runs an SFT training job on Tinker."""

import argparse
import asyncio

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
)


@chz.chz
class GameSeparatedBuilder(ChatDatasetBuilder):
    """Trains on one file and evaluates on a held-out file with no shared games."""

    train_path: str
    val_path: str
    val_size: int

    def __call__(self):
        train_ds, _ = FromConversationFileBuilder(
            common_config=self.common_config, file_path=self.train_path
        )()
        _, val_ds = FromConversationFileBuilder(
            common_config=self.common_config,
            file_path=self.val_path,
            test_size=self.val_size,
        )()
        return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--load-checkpoint", default=None, help="tinker:// state path to resume from")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    renderer_name = args.renderer
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        max_length=8192,
        batch_size=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = GameSeparatedBuilder(
        common_config=common_config,
        train_path=args.data,
        val_path=args.val_data,
        val_size=args.test_size,
    )
    config = chz.Blueprint(train.Config).apply(
        {
            "log_path": args.log_path,
            "recipe_name": "catan_sft",
            "model_name": args.model,
            "renderer_name": renderer_name,
            "dataset_builder": dataset,
            "learning_rate": args.lr,
            "lr_schedule": "linear",
            "lora_rank": args.rank,
            "num_epochs": args.epochs,
            "save_every": args.save_every,
            "eval_every": args.eval_every,
            "wandb_project": "catan-llm",
            "wandb_name": args.run_name,
            **({"load_checkpoint_path": args.load_checkpoint} if args.load_checkpoint else {}),
        }
    ).make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="raise")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
