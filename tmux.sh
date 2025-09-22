#!/bin/bash

# Kill any existing tmux session to start fresh
tmux kill-session -t aerial 2>/dev/null

# Create new session named 'aerial' with first window 'cli'
tmux new-session -d -s aerial -n cli -c /cfs/home/u035679/aerialseg

# Split cli window into 4 equal panes (2x2 grid)
tmux split-window -h -t aerial:cli
tmux select-pane -t aerial:cli.0
tmux split-window -v -t aerial:cli.0
tmux select-pane -t aerial:cli.2
tmux split-window -v -t aerial:cli.2
tmux select-layout -t aerial:cli tiled

# Setup cli window panes
# Top left: vim netrw
tmux send-keys -t aerial:cli.0 'vim ./' Enter

# Top right: git diff ready (pane index 1 after tiling)
tmux send-keys -t aerial:cli.1 'git diff' C-m

# Bottom left: claude command (pane index 2 after tiling)
tmux send-keys -t aerial:cli.2 'claude' Enter

# Bottom right: claude command
tmux send-keys -t aerial:cli.3 'claude' Enter

# Create second window 'gpu' with 4 panes
tmux new-window -t aerial -n gpu -c /cfs/home/u035679/aerialseg

# Split gpu window into 4 equal panes (2x2 grid)
tmux split-window -h -t aerial:gpu
tmux select-pane -t aerial:gpu.0
tmux split-window -v -t aerial:gpu.0
tmux select-pane -t aerial:gpu.2
tmux split-window -v -t aerial:gpu.2

# Setup gpu window panes
# Top left: cd to tex/dissertation and latex compilation ready
tmux send-keys -t aerial:gpu.0 'cd tex/dissertation' Enter
tmux send-keys -t aerial:gpu.0 'latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex'

# Other panes remain in aerialseg directory (default)

# Select the cli window and top-left pane as default
tmux select-window -t aerial:cli
tmux select-pane -t aerial:cli.0

# Attach to the session
tmux attach-session -t aerial