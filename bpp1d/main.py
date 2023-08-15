from enum import Enum
# import json
import commentjson as json
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Sequence
import typer
from typer import Option, Argument
from typing_extensions import Annotated
from bpp3d_dataset.utils.distributions import generate_discrete_dist
from bpp3d_dataset.problems import Problem, Bpp1DRandomInitiator
from bpp1d.models.cg_fit import CGFit
from bpp1d.models.oracle import Oracle

from bpp1d.models.rl.train import RLHyperparam, train_ppo
from bpp1d.structure.solution import Solution
from bpp1d.structure.bin_solution import BinSolution
from bpp1d.utils.exp_parser import ExpModelConfig, ExpProblemConfig
from bpp1d.utils.visualizer import Visualizer
import random
import numpy as np
app = typer.Typer()

rl_app = typer.Typer()
app.add_typer(rl_app, name="rl")

exp_app = typer.Typer()
app.add_typer(exp_app, name="exp")

DEFAULT_RL_TRAIN_ITEMS = [i for i in range(10, 60, 5)]
DEFAULT_RL_CAPACITY = 100
DEFAULT_TRAIN_ENV_NUM = 100
DEFAULT_STEP_PER_EPOCH = 10000

# class DistributionType(str, Enum):
#     uniform = "uniform"
#     normal = "normal"
#     discrete = "discrete"



@rl_app.command("train")
def rl_train(
    distribution_choice: Annotated[str, 
                                    Option("-d", help="Distribution for training environment")] 
                                        = 'uniform',
    items:Annotated[Optional[List[int]], Option("-i", "--item", help="item kinds in the problem")] 
                                = None,
    item_range: Annotated[Optional[List[int]], Option("--ir", "--item-range", help="Item range")] = None,
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
    if item_range:
        items = list(range(item_range[0], item_range[1]))
    
    if not items:
        items = DEFAULT_RL_TRAIN_ITEMS
        

    distribution = generate_discrete_dist(items, distribution_choice, probs=probs)


    target_problem = Problem(Bpp1DRandomInitiator(capacity, item_per_instance, 
                                                    test_instance_num, distribution))
    print(target_problem.configuration)
    param = RLHyperparam(lr, epoch, batch_size, buffer_size, DEFAULT_STEP_PER_EPOCH, DEFAULT_TRAIN_ENV_NUM,
                            item_per_instance, DEFAULT_TRAIN_ENV_NUM, eps_clip, discount_factor)
    res = train_ppo(distribution, capacity, target_problem, model_path,file_name, param)
    print(res)


@exp_app.command("problem")
def experiment_problem(
        experiment_dir: Annotated[Path, Argument(help="experiment directory")] = "experiment/",
        config_filename: Annotated[str, Option("-c", "--config", 
                                                    help="problem configuration file")] = "config.json",
        seed: Annotated[int, Option("-s", "--seed", help="seed")] = 42,
        verbose: Annotated[bool, Option("-v", "--verbose", help="print extra information")] = False,
        visualize: Annotated[bool, Option("--visualize", help="visualize")] = False,
    ):
    
    random.seed(seed)
    np.random.seed(seed)
    
    
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
    true_solutions: Sequence[Solution] = []
    total_infos: Sequence[Dict[str, Any]] = []
    visualize_dir = experiment_dir / "visualize/"
    solution_dir = experiment_dir / "solution/"
    if not solution_dir.exists():
        os.mkdir(solution_dir)

    if visualize:
        if not visualize_dir.exists():
            os.mkdir(visualize_dir)
        # else:
        #     for f in visualize_dir.glob("*"):
        #         os.remove(f)
        
    visualizer = Visualizer(visualize_dir)

    print("===== Solve Problem =====")
    # solve problems
    for i, instance in enumerate(problem):
        print(f"Instance {i}: ")
        models = model_config.create_models(problem.configuration['capacity'], instance)
        true_demands = {i: instance.sequence.count(i) for i in sorted(list(set(instance)))}

        oracle = Oracle(problem.configuration['capacity'], instance, true_demands)
        oracle.name = 'Oracle'
        oracle.build()

        oracle_solution, info = oracle.solve()
        true_solutions.append(oracle_solution)
        with open(solution_dir / f"{i}_Oracle.json", 'w+') as f:
            json.dump(oracle_solution.data_obj, f)

        # print(ground_truth.plan)
        print("Oracle", oracle_solution.metrics)
        results = {}
        infos = {}
        for model in models:
            name = model.name
            model.build()
            solution, info = model.solve()
            results[name] = solution
            infos[name] = info
            if verbose:
                # if isinstance(solution ,BinSolution):
                    # visualizer.visualize_bin_solution(f"{i}_{name}.png", "solution", solution)
                print(f"Model: {name}", solution.metrics)
                with open(solution_dir / f"{i}_{name}.json", 'w+') as f:
                    json.dump(solution.data_obj, f)


        # results['ground_truth'] = true_solution
        if visualize:
            visualize = results.copy()
            visualize['Oracle'] = oracle_solution
            visualizer.visualize_solutions_sns(f"{i}.pdf", "Solution Bin Levels Comparison", 
                                            list(visualize.values()), list(visualize.keys()))


        # all_results['instances'][i] = results
        total_results.append(results)

        if verbose:
            total_infos.append(infos)

    # for i, res in enumerate(total_results):
        # for sol, name in res.items():
            # visualizer.visualize_solutions(f"{i}_{name}.png", "solution", sol)
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
    win_bfs = {}
    all_res = []
    for i, (res, oracle_solution) in enumerate(zip(total_results, true_solutions)):
        # order by waste
        waste = {n:r.waste for n, r in res.items() }
        ordered_waste = sorted([k for k in waste ], key=lambda n: waste[n])
        waste_orders.append(ordered_waste)
        gap = {
                k: {
                    "bins": v.num_bins - oracle_solution.num_bins,
                    "waste": v.waste - oracle_solution.waste,
                    "waste_bin": (v.waste - oracle_solution.waste) / oracle_solution.capacity,
                    "waste_bin_rate": ((v.waste - oracle_solution.waste) / oracle_solution.capacity) 
                                            / oracle_solution.num_bins,
                    "rate": float(v.num_bins) / oracle_solution.num_bins
                }
                for k, v in res.items() 
            }

        all_gap.append(gap)
        
        res_assemble = {
            k: {
                "bins": v.num_bins,
                "waste": v.waste
            }
            for k, v in res.items() 
        }
        all_res.append(res_assemble)
        
        winner = ordered_waste[0] # minimum
        win_rates[winner] = win_rates.get(winner, 0) + 1

        for k in res.keys():
            if ordered_waste.index(k) < ordered_waste.index('best_fit'):
                win_bfs[k] = win_bfs.get(k, 0) + 1
        
        with open(instance_dir / f"instance_{i}.json", 'w+') as f:
            json.dump({n: r.metrics for n, r in res.items()}, f)

        if verbose:
            with open(instance_dir / f"instance_{i}_info.json", 'w+') as f:
                json.dump(total_infos[i], f)
    # gap_avg = [  for g in all_gap]
    gap_avg = {
        k: {
            # "bins": np.average([g[k]["bins"] for g in all_gap]),
            # "waste": np.average([g[k]["waste"] for g in all_gap]),
            "bins": {
                "avg": np.average([g[k]["bins"] for g in all_gap]),
                "std": np.std([g[k]["bins"] for g in all_gap]),
            },

            "waste": {
                "avg": np.average([g[k]["waste"] for g in all_gap]),
                "std": np.std([g[k]["waste"] for g in all_gap]),
            },
            "waste_bin": np.average([g[k]["waste_bin"] for g in all_gap]),
            "waste_bin_rate": np.average([g[k]["waste_bin_rate"] for g in all_gap]),
            "rate": np.average([g[k]["rate"] for g in all_gap]),
        }
        for k in all_gap[0].keys()
    }

    res_avg = {
        k: {
            "bins": {
                "avg": np.average([g[k]["bins"] for g in all_res]),
                "std": np.std([g[k]["bins"] for g in all_res]),
            },

            "waste": {
                "avg": np.average([g[k]["waste"] for g in all_res]),
                "std": np.std([g[k]["waste"] for g in all_res]),
            },
        }
        for k in all_res[0].keys()
    }


    summary = {
                'gap': gap_avg,
                'res': res_avg,
                'wins': win_rates,
                'win_bf': win_bfs,
                'order': waste_orders,
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