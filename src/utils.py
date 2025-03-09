import yaml

def load_config(config_path="config.yaml"):
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# Example usage
if __name__ == "__main__":
    config = load_config()
    print(config)
