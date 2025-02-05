from deepcave.plugins.hyperparameter.ablation_paths import AblationPaths
from deepcave.runs.converters.deepcave import DeepCAVERun
from pathlib import Path

if __name__ == "__main__":
    # Instantiate the run
    run = DeepCAVERun.from_path(Path("logs/DeepCAVE/digits_sklearn/run_1"))

    objective_id1 = run.get_objective_ids()[0]
    objective_id2 = None  # replace with run.get_objective_ids()[1] for multi-objective importance
    budget_id = run.get_budget_ids()[0]
    
    default_config = run.configspace.get_default_configuration()
    print(run.encode_config(default_config)[0], default_config.get_array()[0])
    
    assert run.encode_config(default_config)[0] == default_config.get_array()[0]