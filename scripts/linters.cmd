@echo off

pushd ..

echo.
echo Format using isort
isort convince --sp=.isort.cfg
isort tests --sp=.isort.cfg

echo.
echo Format using black
black -q convince --config=pyproject.toml
black -q tests --config=pyproject.toml

echo.
echo Validate using flake8
flake8 convince --config=.flake8
flake8 tests --config=.flake8

popd
