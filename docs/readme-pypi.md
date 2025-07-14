# ideas to set up pypi packages

## creation

- create a pypi account like `cancerdata-packages`
- we both should have access to that account and email adresses registered
- structure proposal
  - package name like: `recoding`
    - file name: `create_substance_service_var`
    - file name: `create_protocol_service_var`
- package call:

```python
from recoding import create_substance_service_var
from recoding import create_protocol_service_var

# example, df_raw must have been set up already
df_subst = create_substance_service_var(df_raw)
```

## workflow

- github repo owner has a token from pypi account stored
- on repo change:
  - `.toml` version increments
  - changes are committed to github
  - new local build is generated (buildtools)
  - local build is pushed to pypi using token (twine)

## hints

- token is stored in the file `~/.pypirc` after first usage of twine

- one time setup

```python
# dont use uv add here since this is only for package setup
uv pip install twine hatchling build
```

- on repo change

```python
# rebuild and upload in one step. pypi is the entry in .pypirc file!
rm -r dist/ & py -m build && twine upload -r pypi dist/*
```

