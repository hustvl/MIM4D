
#!/usr/bin/env bash

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CONFIG=projects/configs/MIM4D/uvtr_convnext_s_vs0.075_finetune.py
CHECKPOINT=ckpts/uvtrs_mim4d_vs0.075/uvtrs_mim4d_vs0.075_finetune.pth


python3 $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval bbox