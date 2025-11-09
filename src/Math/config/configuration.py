from src.Math.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.Math.utils.common import read_yaml
from pathlib import Path
from src.Math.entity.config_entity import DataIngestion, DataStoring


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_ingestion_config(self) -> DataIngestion:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestion(
            docs=Path(config.data_folder),
            EMBED_MODEL=config.EMBED_MODEL,
        )
        return data_ingestion_config

    def get_data_storing_params(self) -> DataStoring:
        params = self.params.data_storing

        data_storing_params = DataStoring(
            dimension=params.dimension,
            collection_name=params.collection_name,
            contexts=params.contexts,
            sources=params.sources,
        )
        return data_storing_params
