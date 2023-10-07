#!/bin/bash

cd ..

# Format using isort
echo "Format using isort"
isort convince --sp=.isort.cfg
isort tests --sp=.isort.cfg

# Format using black
echo ""
echo "Format using black"
black -q convince --config=pyproject.toml
black -q tests --config=pyproject.toml

# Validate using flake8
echo ""
echo "Validate using flake8"
flake8 convince --config=.flake8
flake8 tests --config=.flake8

# Change back to the original directory
cd -
