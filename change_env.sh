#!/usr/bin/env bash
set -e
set -x

source activate demo
cd /app
python /app/app.py