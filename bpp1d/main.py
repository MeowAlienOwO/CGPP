from enum import Enum
import json
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Sequence
import typer
from typer import Option, Argument
from typing_extensions import Annotated
from bpp3d_dataset.utils.distributions import Discrete, Uniform, Binomial
from bpp3d_dataset.problems import Problem, Bpp1DRandomInitiator
from bpp1d.models.cg_fit import CGFit

from bpp1d.models.rl.train import RLHyperparam, train_ppo
from bpp1d.structure.solution import Solution
from bpp1d.utils.exp_parser import ExpModelConfig, ExpProblemConfig

app = typer.Typer()

rl_app = typer.Typer()
app.add_typer(rl_app, name="rl")

exp_app = typer.Typer()
app.add_typer(exp_app, name="exp")

DEFAULT_RL_TRAIN_ITEMS = [i for i in range(10, 60, 5)]
DEFAULT_RL_CAPACITY = 100
DEFAULT_TRAIN_ENV_NUM = 100
DEFAULT_STEP_PER_EPOCH = 10000

class DistributionType(str, Enum):
    uniform = "uniform"
    normal = "normal"
    discrete = "discrete"



@rl_app.command("train")
def rl_train(
    distribution_choice: Annotated[DistributionType, 
                                    Option("-d", help="Distribution for training environment")] 
                                        = DistributionType.uniform,
    items:Annotated[List[int], Option("-i", "--item", help="item kinds in the problem")] 
                                = DEFAULT_RL_TRAIN_ITEMS,
    probs:Annotated[Optional[List[float]], Option("-p", "--prob", 
                                                        help="probability list")] = None,
    capacity: Annotated[int, Option("-C", help="Bin capacity")] = DEFAULT_RL_CAPACITY,
    model_path: Annotated[Path, Option("--path", help="Model path")] = Path("models/"),
    file_name: Annotated[str, Option("--file", help="Model filename")] = "policy.pth",
    epoch: Annotated[int, Option(help="Train total epoch")] = 1000,
    lr: Annotated[float, Option("--lr", help="Learning Rate")] = 1e-3, 
    batch_size: Annotated[int, Option(help="Batch size")] = 128,
    buffer_size: Annotated[int, Option(help="Bin size")] = 20000,
    item_per_instance: Annotated[int, Option("--item-num", help="Number of items in instance")] = 1000,
    test_instance_num: Annotated[int, Option("--test-num", help="Number of instance in test")] = 100,
    eps_clip: Annotated[float, Option(help="PPO Hyperparameter: clip factor")] = 0.2,
    discount_factor: Annotated[float, Option(help="PPO Hyperparameter")] = 0.095,
):

    if not os.path.exists(model_path):
        os.mkdir(model_path)


    if distribution_choice == DistributionType.uniform:
        distribution = Uniform(items)
    elif distribution_choice == DistributionType.normal:
        distribution = Binomial(items)
    elif distribution_choice == DistributionType.discrete:
        distribution = Discrete(probs, items)
    else:
        print("Not implemented")
        raise NotImplementedError

    # target_problem = make_bpp(distribution_choice.capitalize() + "1D")
    target_problem = Problem(Bpp1DRandomInitiator(capacity, item_per_instance, 
                                                    test_instance_num, distribution))
    param = RLHyperparam(lr, epoch, batch_size, buffer_size, DEFAULT_STEP_PER_EPOCH, DEFAULT_TRAIN_ENV_NUM,
                            item_per_instance, DEFAULT_TRAIN_ENV_NUM, eps_clip, discount_factor)
    res = train_ppo(distribution, capacity, target_problem, model_path,file_name, param)
    print(res)


@exp_app.command("problem")
def experiment_problem(
        experiment_dir: Annotated[Path, Argument(help="experiment directory")] = "experiment/",
        config_filename: Annotated[str, Option("-c", "--config", 
                                                    help="problem configuration file")] = "config.json",
        verbose: Annotated[bool, Option("-v", "--verbose", help="print extra information")] = False
    ):
    
    config_file = experiment_dir / config_filename
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not exists: {config_file}")
    
    with open(config_file) as f:
        config = json.load(f)

    print("===== Start =====")
    model_config = ExpModelConfig(config['models'])
    problem_config = ExpProblemConfig(config['problem'])
    problem = problem_config.create_problem()

    print("Problem Configuration: ", problem.configuration)
    print("Models: ", model_config.models)

    total_results: Sequence[Dict[str, Solution]] = []
    total_infos: Sequence[Dict[str, Any]] = []

    print("===== Solve Problem =====")
    # solve problems
    for i, instance in enumerate(problem):
        print(f"Instance {i}: ")
        models = model_config.create_models(problem.configuration['capacity'], instance)
        true_demands = {i: instance.sequence.count(i) for i in sorted(list(set(instance)))}
        ground_truth = CGFit(problem.configuration['capacity'], instance, true_demands)
        ground_truth.name = 'ground_truth'
        models.append(ground_truth)
        results = {}
        infos = {}
        for model in models:
            name = model.name
            # if verbose:
                # print(f"Model {model.name} executing.")
            model.build()
            solution, info = model.solve()
            results[name] = solution
            infos[name] = info
            if verbose:
                print(f"Model: {name}", solution.metrics)

        # all_results['instances'][i] = results
        total_results.append(results)

        if verbose:
            total_infos.append(infos)


    print("===== Analysis =====")
    
    instance_dir = experiment_dir / 'instances/'
    # if instance_dir.exists() and instance_dir.is_dir():
    #     # clear instances
    #     os.rmdir(instance_dir)
    if not instance_dir.exists():
        os.mkdir(instance_dir)

    # analyse and print results
    win_rates = {}
    all_gap = []
    waste_orders = []
    for i, res in enumerate(total_results):
        # order by waste
        waste = {n:r.waste for n, r in res.items() }
        ordered_waste = sorted([k for k in waste if k != 'ground_truth'], key=lambda n: waste[n])
        waste_orders.append(ordered_waste)
        gap = {k:v.num_bins / res['ground_truth'].num_bins  
                for k, v in res.items() if k != 'ground_truth'}
        all_gap.append(gap)
        
        winner = ordered_waste[0] # minimum
        win_rates[winner] = win_rates.get(winner, 0) + 1
        
        with open(instance_dir / f"instance_{i}.json", 'w+') as f:
            json.dump({n: r.metrics for n, r in res.items()}, f)

        if verbose:
            with open(instance_dir / f"instance_{i}_info.json", 'w+') as f:
                json.dump(total_infos[i], f)

    # avg_gap = [for gap in all_gap]

    summary = {
                'wins': win_rates,
                'order': waste_orders,
                'gap': gap
        }
    if verbose:
        print(summary)

    with open(experiment_dir / 'result.json', "w+") as f:
        json.dump(summary, f, indent=2)

    print("===== Finished =====")


# @exp_app.command("interactive")
# def experiment_interactive():
#     raise NotImplementedError

if __name__ == "__main__":
    app()