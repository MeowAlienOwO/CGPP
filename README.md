# Column Generation Plan-and-Pack

This repository is for the paper *Pattern based learning through pricing for the bin packing problem*



## Abstract

Pattern is a popular form of knowledge and experience. Discovering and reusing high quality patterns has long been a main task in many data mining applications. In practical bin packing problems, mostly NP-Hard, the size of the next item to be packed can be unknown in advance. Many existing methods tend to reuse some example packing patterns that worked well in previous experiences. However, when the problem conditions (e.g. distributions of item sizes) change, the patterns that performed well for the previous circumstances can become less effective and adoption of these patterns would lead to poor sub-optimal solutions. It is, therefore, crucial to establish clear relationships between the problem solving circumstances and the quality levels of various patterns. Inspired by the dualism and the concept of shadow price in integer programming, in this research, we propose an iterative scheme to accurately quantify the value of patterns under each particular condition and then dynamically generate the ``best'' patterns based on the latest forecast of the problem condition. Our simulation results show that the proposed algorithm significantly outperforms existing heuristic methods when tackling online bin packing with finite discrete item types. The proposed method is generalisable for several other online combinatorial optimisation problems with linear/integer programming formulations. 

# How to use


## Prerequisite
This project is depending on the [bpp3d-dataset](https://github.com/MeowAlienOwO/Bpp3dDataset) package for datasets and problem definition.


## Install
Environment set up

```bash
conda create --name env_name --file conda-linux-64.lock
conda activate env_name
poetry install 

```

## Command Line Interface

Execute experiment: 
```bash
python -m bpp1d.main exp problem your/experiment/directory/ -v --visualize

```

A `config.json` must be put under the experiment folder to define the problem and model parameters. 
An example configuration json file: 

```json
{
    "problem": "Uniform-1D", // should match problem name in bpp3d-dataset
    "models":{
        "best_fit": { 
            "type": "heuristic",
            "heuristic": "best_fit"
        },
        "rl_model": {
            "type": "rl_model",
            "checkpoint_path": "your/rl/checkpoint.pth"
        },
        "cg_shift": { // our model, due to historical reasons it is still cg_shift but will be updated in later release
            "type": "cg_shift",
            "priori": {
                "name": "uniform"
            },
            "memory_size": 100,
            "section_size": 300,
            "underestimate_tolerance": 5,
            "overestimate_tolerance": 1
        }
    }

}
```

Train RL:

```bash

python -m bpp1d.main rl train -d distribution-name --path rl/model/path --epoch 501
```