#!/bin/bash

tensorboard --logdir=./results/tensorboard &
firefox "http://localhost:6006/" 