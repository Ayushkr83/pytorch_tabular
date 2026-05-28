import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, "/mnt/hdd2/ayush/personal/pytorch_tabular/src")
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
import pytorch_tabular.models as models
# A simple mapping for user convenience
MODEL_CONFIG_MAP = {
    "CategoryEmbedding": "CategoryEmbeddingModelConfig",
    "DANet": "DANetConfig",
    "NODE": "NodeConfig",
    "TabNet": "TabNetModelConfig",
    "MDN": "MDNConfig",
    "AutoInt": "AutoIntConfig",
    "TabTransformer": "TabTransformerConfig",
    "FTTransformer": "FTTransformerConfig",
    "GATE": "GatedAdditiveTreeEnsembleConfig",
    "GANDALF": "GANDALFConfig"
}

def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(yaml_path, data_path):
    config = load_config_from_yaml(yaml_path)

    print(f"Loading data from {data_path}...")
    # Adjust this based on your dataset; here we assume a standard CSV with a header
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # Optional: Fill NaNs or preprocess if necessary
    
    train, test = train_test_split(df, random_state=42)
    train, val = train_test_split(train, random_state=42)

    # 1. Data Config
    data_config = DataConfig(**config.get('data_config', {}))

    # 2. Trainer Config
    trainer_config = TrainerConfig(**config.get('trainer_config', {}))

    # 3. Optimizer Config
    optimizer_config = OptimizerConfig(**config.get('optimizer_config', {}))

    # 4. Model Config
    model_cfg = config.get('model_config', {})
    model_type = model_cfg.pop('model_type', 'DANet')
    
    config_class_name = MODEL_CONFIG_MAP.get(model_type, model_type)
    
    if model_type == "TabM":
        from tabm_model import TabMConfig, TabMModel
        model_config = TabMConfig(**model_cfg)
        model_callable = TabMModel
    else:
        if not hasattr(models, config_class_name):
            raise ValueError(f"Model config {config_class_name} not found in pytorch_tabular.models")
        ModelConfigClass = getattr(models, config_class_name)
        model_config = ModelConfigClass(**model_cfg)
        model_callable = None

    print(f"Initializing TabularModel with {model_type}...")
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        model_callable=model_callable,
        verbose=True
    )

    print("Training the model...")
    tabular_model.fit(train=train, validation=val)

    print("Evaluating the model on test set...")
    result = tabular_model.evaluate(test)
    print("Evaluation Results:", result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch Tabular Models from YAML")
    parser.add_argument("--config", type=str, required=True, help="Path to the master YAML config file")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset CSV file")
    
    args = parser.parse_args()
    main(args.config, args.data)
