#!/bin/bash

# 自己隨便設定一個session名稱
SESSION_NAME="single_miou"
PICTURE=501
# 檢查這個session本來是否存在
tmux has-session -t ${SESSION_NAME} 2>/dev/null
if [ $? != 0 ]; then
    # 先開啟新的session
    # shell 0
    tmux new-session -s ${SESSION_NAME} -n bash -d
    tmux send-keys -t ${SESSION_NAME}:0 'conda activate ml2022' C-m
    tmux send-keys -t ${SESSION_NAME}:0 'cd /home/frank/Desktop/nv/ML_Final_Project' C-m
    tmux send-keys -t ${SESSION_NAME}:0 'python3 test_miou.py --file_path tflite/0501.tflite --test_batch 130 ' C-m

    # 已經有session了所以使用new-window
    # shell 1
    tmux new-window -n bash -t ${SESSION_NAME}
    tmux send-keys -t ${SESSION_NAME}:1 'conda activate ml2022' C-m
    tmux send-keys -t ${SESSION_NAME}:1 'cd /home/frank/Desktop/nv/ML_Final_Project' C-m
    tmux send-keys -t ${SESSION_NAME}:1 'python3 test_miou.py --file_path tflite/0507.tflite --test_batch 130 ' C-m

    # shell 2
    tmux new-window -n bash -t ${SESSION_NAME}
    tmux send-keys -t ${SESSION_NAME}:2 'conda activate ml2022' C-m
    tmux send-keys -t ${SESSION_NAME}:2 'cd /home/frank/Desktop/nv/ML_Final_Project' C-m
    tmux send-keys -t ${SESSION_NAME}:2 'python3 test_miou.py --file_path tflite/0510.tflite --test_batch 130 ' C-m

    # shell 3
    tmux new-window -n bash -t ${SESSION_NAME}
    tmux send-keys -t ${SESSION_NAME}:3 'conda activate ml2022' C-m
    tmux send-keys -t ${SESSION_NAME}:3 'cd /home/frank/Desktop/nv/ML_Final_Project' C-m
    tmux send-keys -t ${SESSION_NAME}:3 'python3 test_miou.py --file_path tflite/0521.tflite --test_batch 130 ' C-m

    # shell 4
    tmux new-window -n bash -t ${SESSION_NAME}
    tmux send-keys -t ${SESSION_NAME}:4 'conda activate ml2022' C-m
    tmux send-keys -t ${SESSION_NAME}:4 'cd /home/frank/Desktop/nv/ML_Final_Project' C-m
    tmux send-keys -t ${SESSION_NAME}:4 'python3 test_miou.py --file_path tflite/0528.tflite --test_batch 130 ' C-m
    # shell 5
    tmux new-window -n bash -t ${SESSION_NAME}
    tmux send-keys -t ${SESSION_NAME}:5 'conda activate ml2022' C-m
    tmux send-keys -t ${SESSION_NAME}:5 'cd /home/frank/Desktop/nv/ML_Final_Project' C-m
    tmux send-keys -t ${SESSION_NAME}:5 'python3 test_miou.py --file_path tflite/0532.tflite --test_batch 130 ' C-m

    # 將畫面切回一開始的window
    tmux select-window -t ${SESSION_NAME}:0
fi

tmux attach-session -t ${SESSION_NAME}