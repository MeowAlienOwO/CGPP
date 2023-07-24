from pathlib import Path
import json
from typing import Dict, List, Sequence
from bpp1d.models import VALID_MODELS, Model, RLModel, CGFit, CGReplan, CGStateShift
from bpp1d.models.heruistics import HeuristicModel
from bpp1d.utils.heuristic_choice import generate_heuristic
from bpp3d_dataset.utils.distributions import Discrete, Uniform, Binomial, Poisson, generate_discrete_dist
from bpp3d_dataset.problems import Problem, make_bpp, Bpp1DRandomInitiator

from bpp1d.utils.state_estimator import KernelDensityEstimator, SimpleStateEstimator


class BaseConfig:
    def __init__(self, content:Path | str | Dict) -> None:
        """"Parse  configuration given file, json string or dictionary 
        The format of configuration dictionary is as follows:

        {
            "model_name/problem_name": {
                "param1": ...
                "param2": ...
            }
        }

        Args:
            content (Path | str | Dict): content of config
        """
        self._raw_content = content
        if isinstance(content, Path):
            with open(content) as f:
                self._config_dict = json.load(f)
        elif isinstance(content, str):
            try:
                self._config_dict = json.loads(content)
            except ValueError:
                # use _raw_content as a string
                self._config_dict = {}
                # self.predefined_name = content
        else:
            self._config_dict = content
    @property
    def config(self) -> Dict | str:
        if not self._config_dict:
            return str(self._raw_content)
        else:
            return self._config_dict




class ExpModelConfig(BaseConfig):
    # TODO: predefine models
    def __init__(self, content: Path | str | Dict) -> None:
        super().__init__(content)
    
    @property
    def models(self) -> List[str]:
        return [k for k in self._config_dict.keys()]

    def get_config(self, key: str) -> Dict:
        return self._config_dict[key]
    
    def create_models(self, capacity: int, instance: Sequence[int]) -> List[Model]:
        return [self._create_single(n, capacity, instance) for n in self.models]

    def _create_single(self, model_name: str, capacity: int, instance: Sequence[int]) -> Model:
        config = self.get_config(model_name)
        model_type = config['type']
        if model_type == 'heuristic':
            choice_fn = generate_heuristic(config['heuristic'])
            return HeuristicModel(capacity, instance, name=model_name, choice_fn= choice_fn)
        elif model_type == 'rl_model':
            return RLModel(capacity, instance, config['checkpoint_path'])
        elif model_type == 'cg_fit':
            # demand = config.get('demand', {i: instance.count(i) 
                                            # for i in sorted(list(set(instance)))})

            demand = {int(k): v for k ,v in config.get('demand', {}).items()}
            
            return CGFit(capacity, instance, demand, name=model_name)

        elif model_type == 'cg_replan':
            dist = config['priori']
            # items = sorted(set(instance))
            # distribution = self._generate_distribution(dist, items)
            items = self._generate_item_size(dist, instance)
            distribution = generate_discrete_dist(dist_key=dist['name'], items=items, kwargs=dist)
            # print(distribution)
            
            return CGReplan(capacity, instance, distribution, consider_opened_bins=True)

        elif model_type == 'cg_shift':
            dist = config['priori']
            # items = sorted(set(instance))
            items = self._generate_item_size(dist, instance)
            distribution = generate_discrete_dist(dist_key=dist['name'], items=items, kwargs=dist)
            # distribution = self._generate_distribution(dist, items)
            if 'estimator' in config:
                if config['estimator'] == 'kernel':
                    state_estimator = KernelDensityEstimator(distribution)
                else:
                    state_estimator = SimpleStateEstimator(distribution)
            else:
                state_estimator = SimpleStateEstimator(distribution)
            # print(distribution)
            return CGStateShift(capacity, instance, state_estimator, consider_opened_bins=True)
        else:
            raise NotImplementedError(f"Model {model_type} is not implemented")
    
    def _generate_item_size(self, dist: Dict, instance: Sequence[int]):
        if 'items' in dist:
            if isinstance(dist['items'], List):
                # list of item size
                return sorted(dist['items'])
            elif isinstance(dist['items'], Dict):
                # define start, end, step to generate items
                i,j,s = dist['items'].get('start', 0), dist['items'].get('end'), dist['items'].get('step', 1)
                return list(range(i, j, s))
            else:
                raise NotImplementedError(f"Item definition not implemented: {dist}")
        else:
            return sorted(list(set(instance)))



class ExpProblemConfig(BaseConfig):
    def __init__(self, content: Path | str | Dict) -> None:
        super().__init__(content)
    
    def create_problem(self) -> Problem:
        if not self._config_dict:
            return make_bpp(self._raw_content)
        else:
            # assume not predefined problem
            initiator = Bpp1DRandomInitiator(capacity=self.config['capacity'],
                                                item_num=self.config['item_num'],
                                                instance_num=self.config['instance_num'],
                                                distribution=generate_discrete_dist(**self.config['distribution'])
                                            )
            return Problem(initiator, None)
