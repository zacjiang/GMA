#!/usr/bin/env bash
python evaluate.py --model checkpoints/gma-chairs.pth --dataset chairs
python evaluate.py --model checkpoints/gma-things.pth --dataset sintel
python evaluate.py --model checkpoints/gma-sintel.pth --dataset sintel
python evaluate.py --model checkpoints/gma-kitti.pth --dataset kitti