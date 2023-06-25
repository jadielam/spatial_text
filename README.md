# Spatial Text
Library with multiple text-spatial algorithms that can be useful to derive layout information from
text in the 2D space.

## Testing
Simply run `pytest`


## Using pre-commit hooks
Run

`pre-commit install --hook-type pre-commit --hook-type pre-push`

The above command installs the precommit hooks in the repo, usually in the `.git/hook/pre-commit` directory

If you wish to run the hook manually on every file, you can use this command:

`pre-commit run --all-files`
