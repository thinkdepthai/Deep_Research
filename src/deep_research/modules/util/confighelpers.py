import os
import yaml


def get_config_yml(path, section_name, subsection_name=None):
    """Return a given section of a YAML file, optionally its subsection."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, encoding="utf8") as f:
        data = yaml.safe_load(f)
        try:
            return (
                data[section_name]
                if subsection_name is None
                else data[section_name][subsection_name]
            )
        except KeyError as e:
            raise KeyError(
                f"No such section or subsection in config file: {section_name}, {subsection_name}. Config file: {path}"
            ) from e


def load_config(stage_name=None):
    """Load configuration based on the given stage name, or from environment variable."""
    stage_name = stage_name or os.environ.get("STAGE")
    CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yml")

    return get_config_yml(
        path=CONFIG_PATH, section_name="stages", subsection_name=stage_name
    )
