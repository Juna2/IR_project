cd ..
current_filename=${0##*/}
tag=$(echo $current_filename| cut -d'_' -f 2)
tag=$(echo $tag| cut -d'.' -f 1)
current_ip=$(curl ifconfig.co)
if [ -z "$current_ip" ]
then
      current_ip="?.?.?.?"
fi


seed_list=(0)
trial_list=(0)
gpu_num=0
for seed in ${seed_list[@]}
do
    for trial in ${trial_list[@]}
    do
        python main.py --result_path_tag $tag --ip $current_ip --trial $trial --model vgg16 --gpu $gpu_num --epochs 2 --lr 0.0001 --lambda_for_final 1000000 --batch_size 8 --mask_data_size 56 --target_layer features_10 --dataset_class HAM10000Dataset --data_path /home/juna/dataset/HAM10000/original --metrics IoU --train_func train_autograd --val_func validation_autograd --convert_model_from same_kind --IoU_func get_IoU_all_pos_autograd_compl --optimizer adam --only_upd_fc False --R_process max_norm --seed $seed --no_healthy_pat False --IoU_compact_save False --loss_type mask_LRP_seg --loss_filter all --interpreter grad_cam --what_to_patch_on_val malignant --use_autograd True --mask_ratio 0.05 --small_dataset True
    done
done


python ./utils/alarm.py $tag $current_ip $gpu_num
cd execute
bash execute_$(echo $tag)-next.sh















































