#!/bin/bash
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
python3 -m venv .venv
source .venv/bin/activate
python --version
pip install --upgrade pip
pip install -r requirements.txt
