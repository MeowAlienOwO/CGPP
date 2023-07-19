from pathlib import Path
import json
from typing import Dict, List, Sequence
from bpp1d.models import VALID_MODELS, Model, RLModel, CGFit, CGReplan, CGStateShift
from bpp1d.models.heruistics import HeuristicModel
from bpp1d.utils.heuristic_choice import generate_heuristic
from bpp3d_dataset.utils.distributions import Discrete, Uniform, Binomial, Poisson, generate_discrete
from bpp3d_dataset.problems import Problem, make_bpp, Bpp1DRandomInitiator

from bpp1d.utils.state_estimator import SimpleStateEstimator


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
        return [k for k in self._config_dict.keys() if k in VALID_MODELS]

    def get_config(self, key: str) -> Dict:
        return self._config_dict[key]
    
    def create_models(self, capacity: int, instance: Sequence[int]) -> List[Model]:
        return [self._create_single(n, capacity, instance) for n in self.models]

    def _create_single(self, model_name: str, capacity: int, instance: Sequence[int]) -> Model:
        config = self.get_config(model_name)
        if model_name == 'heuristic':
            choice_fn = generate_heuristic(config['type'])
            return HeuristicModel(capacity, instance, name=config['type'], choice_fn= choice_fn)
        elif model_name == 'rl_model':
            return RLModel(capacity, instance, config['checkpoint_path'])
        elif model_name == 'cg_fit':
            demand = config.get('demand', {i: instance.count(i) 
                                            for i in sorted(list(set(instance)))})
            return CGFit(capacity, instance, demand)

        elif model_name == 'cg_replan':
            dist = config['priori']
            items = sorted(set(instance))
            distribution = self._generate_distribution(dist, items)
            
            return CGReplan(capacity, instance, distribution, consider_opened_bins=True)

        elif model_name == 'cg_shift':
            dist = config['priori']
            items = sorted(set(instance))
            distribution = self._generate_distribution(dist, items)
            state_estimator = SimpleStateEstimator(distribution)
            return CGStateShift(capacity, instance, state_estimator, consider_opened_bins=True)
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented")

    def _generate_distribution(self, dist: Dict, items: Sequence[int]) -> Discrete:
        dist_name = dist['name']
        if dist_name == 'uniform':
            return Uniform(items)
        elif dist_name == 'normal':
            p = dist.get('p', 0.5)
            return Binomial(items, p)
        elif dist_name == 'poisson':
            mu = dist.get('mu', 0.6)
            return Poisson(items, mu)
        elif dist_name == 'discrete':
            items = dist.get('items', items)
            probs = dist['probs']
            return Discrete(probs, items)
        else:
            raise NotImplementedError(f"Distribution {dist_name} is not implemented")

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
                                                distribution=generate_discrete(**self.config['distribution'])
                                            )
            return Problem(initiator, None)
# 


# def parse_model_config_file(file: Path) -> ExpModelConfig:
#     """A simple wrapper of ExpModelConfig

#     Args:
#         file (Path): configuration file

#     Returns:
#         ExpModelConfig: configuration object
#     """

#     return ExpModelConfig(file)