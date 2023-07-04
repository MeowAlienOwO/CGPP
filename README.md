# Pattern Based 1dbpp



# How to use

## Install

```bash

python -m pip install .
```


## Dev notes

Environment stack:
- conda: virtual env support, provide basic calculation libs(torch, etc)
    - Q: torch-cuda and torch-cpu need to be separated, how to manage?
- poetry: package manager for most of used libs
- tox: manage testing(typing, linting) with poetry, **do not** create extra virtual env

