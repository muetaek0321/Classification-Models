from argparse import ArgumentParser, Namespace


__all__ = ["make_train_parser"]


def make_train_parser() -> Namespace:
    """コマンドライン引数の設定
    """
    parser = ArgumentParser()
    
    parser.add_argument(
        "-c", "--config", type=str, default="./config/train_config.toml", help="設定ファイルのパス"
    )
    parser.add_argument(
        "-r", "--resume", type=str, default=None, help="学習再開するモデルのパス"
    )
    
    args = parser.parse_args()
    
    return args
