#!/bin/bash
# 自己隨便設定一個session名稱
SESSION_NAME="single_quant"
PICTURE=501
# 檢查這個session本來是否存在
tmux has-session -t ${SESSION_NAME} 2>/dev/null
if [ $? != 0 ]; then
    # 先開啟新的session
    # shell 0
    tmux new-session -s ${SESSION_NAME} -n bash -d
    tmux send-keys -t ${SESSION_NAME}:0 'conda activate ml2022' C-m
    tmux send-keys -t ${SESSION_NAME}:0 'cd /home/frank/Desktop/nv/ML_Final_Project' C-m
    tmux send-keys -t ${SESSION_NAME}:0 "python single_quantization.py --output_name 0$PICTURE.tflite --glob 0$PICTURE" C-m

    # 已經有session了所以使用new-window
    # shell 1
    for (( num=1; num<=35; num++ ))
    # for (( num=36; num<=70; num++ ))
    # for (( num=71; num<=105; num++ ))
    # for (( num=106; num<=129; num++ ))
    do
        Current_PICTURE=$(($PICTURE+$num))
        tmux new-window -n bash -t ${SESSION_NAME}
        tmux send-keys -t ${SESSION_NAME}:$num 'conda activate ml2022' C-m
        tmux send-keys -t ${SESSION_NAME}:$num 'cd /home/frank/Desktop/nv/ML_Final_Project' C-m
        tmux send-keys -t ${SESSION_NAME}:$num "python single_quantization.py --output_name 0$Current_PICTURE.tflite --glob 0$Current_PICTURE" C-m
    done
    # 將畫面切回一開始的window
    tmux select-window -t ${SESSION_NAME}:0
fi

tmux attach-session -t ${SESSION_NAME}

1 7 10 21 28 32