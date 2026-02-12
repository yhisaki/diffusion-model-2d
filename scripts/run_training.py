import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config("config.yaml")
    print(config)


if __name__ == "__main__":
    main()
