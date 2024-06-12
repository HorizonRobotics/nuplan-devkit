# The NuPlan 1.?

This is a custom-maintained version of the NuPlan. We make some changes based on the official NuPlan-devkit and make it easier for R&D purposes.

## New Features

### Conditional yaml formatting

In the old version, you can only use explicit values in yaml config, such as `num_nodes: 1` or `path: ${cfg.output_dir}`, which brings some inconveniences. What if 
we want to config a value based on some other values, perform simple arithmetic operations? 

With the added `eval` resolver for `OmegaConf`, now you can type such conditional formatting in yaml. For example:

```yaml
input_elements: {$eval:'2*3'}   # input_elements: 6 
```

To pass in primitive data structures, wrap them up in a `DictConfig` or a `ListConfig`:

```yaml
input_elements_list: ${eval:'ListConfig([2.0]*2+[0.5]*2+[0.0]*2)'}  # input_elements_list: ListConfig([2.0, 2.0, 0.5, 0.5, 0.0, 0.0])
input_elements_dict: ${eval:'DictConfig({1:2.5})'} # input_elements_dict: DictConfig({1: 2.5})
```

You can also use `if...else...` if you want to base one value on another value. 

### Load scenarios faster

We include 2 new ways to load scenarios faster.

#### Method 1: Use pickled scenario file

For `NuplanScenarioBuilder`, it now supports loading a list of scenario objects packed inside a single pickle file. This is useful when you just dumped some secific scenarios on the disk and want to examine 
them further or use for overfitting, etc. 

Just put the path to the file in `senario_builder.scenario_pickle_path` and it is done. You still need to compute features, though.

#### Method 2: Use cached .csv metadata

If you've cached your training features and targets with NuPlan's caching function, then you can use the `.csv` metadata file generated
at the end of the caching and load features and targets directly.

Put the path to the CSV file in `cache.cache_metadata_path` and you're good to go. This will return a list of `CachedScenario` objects, different from method 1.

### Revamped checkpoint loading behavior

The old version has a "resume training" function, but we don't find it very useful. So we added some new configs in the `default_training.yaml`:

```yaml
# Pre-train checkpoint
checkpoint:                                 # Specify your training pre-train checkpoint here
    ckpt_path: null                             # Path to the checkpoint file
    strict: true                                # Checkpoint loading mode. true will make sure there are no unexpected or missing weights
    resume: false                               # Whether to resume training from the ckpt_path
```

Whether you want to resume training or use pre-trained weights and start new training, they will be better handled here.

### Nested objective and metric

Objectives and metrics are now wrapped by a `FlatDict` object which flattens nested dictionaries. This means you can define objective or metric classes that 
returns multiple values, not just a single one.

Example:

```python
class MyObjective(AbstractObjective):
    ...
    @property
    def name(self):
        return "my_objective"

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        ...
        return {"safety_loss": loss_safety, "comfort_loss": {"longitudinal": comfort_long, "lateral": comfort_lat}}

# In Tensorboard, they will be logged as "my_objective.safety_loss", "my_objective.comfort_loss.longitudinal" and "my_objective.comfort_loss.lateral"
```

## Latest Updates (2024.06)

* Upgraded pytorch-lightning to 2.2.5, with code changes that supports the following:
    * the latest `lightning.Trainer` class API;
    * efficient initialization (for training only) https://lightning.ai/docs/pytorch/stable/advanced/model_init.html;
* Upgraded hydra to 1.3.2
* Removed `cfg.gpu` key since it is not actively used and would cause confusion;
* Added support for conditional yaml formatting.
* Added `scenario_pickle_path` to `NuPlanScenarioBuilder` class.
* Put all requirements into one single requirements.txt. Note that PyTorch is not specified here.
* Added `FlatDict` to resolve nested metrics and objectives.