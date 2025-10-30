#!/bin/bash
DATASET="MMIU"
MODEL="llava_one_vision"

declare -A domain_tasks

# MUIR
# domain_tasks["All"]="Geographic_Understanding Diagram_Understanding Image_Text_Matching Difference_Spotting Visual_Retrieval Counting Attribute_Similarity Scene_Understanding Action_Understanding Visual_Grounding Cartoon_Understanding Ordering"

# MMIU
domain_tasks["2D-spatial"]="Homography_estimation Image_Captioning_with_Spatial_Context Image_Spatial_Transformation_Estimation jigsaw_puzzle_solving point_tracking ravens_progressive_matrices single_object_tracking Icon_Question_Answering_with_Spatial_Context"
# domain_tasks["3D-spatial"]="Multiview_reasoning threed_cad_recognition"
# domain_tasks["Continuous-temporal"]="next_img_prediction meme_vedio_understanding temporal_ordering"
# domain_tasks["Discrete-temporal"]="gui_next_action_prediction textual_cloze visual_cloze visual_coherence visual_ordering"
# domain_tasks["High-level-obj-semantic"]="visually_grounded_reasoning_nlvr2"
domain_tasks["3D-spatial"]="Multiview_reasoning threed_cad_recognition threeD_Depth_Estimation threed_indoor_recognition threeD_Object_Detection threeD_Object_Tracking threeD_Pose_Estimation threeD_question_answering threeD_Scene_Reconstruction"
domain_tasks["Discrete-temporal"]="gui_next_action_prediction textual_cloze visual_cloze visual_coherence visual_ordering"
domain_tasks["Continuous-temporal"]="next_img_prediction meme_vedio_understanding temporal_ordering action_quality_assessment general_action_recognition mevis temporal_localization video_captioning"
domain_tasks["High-level-obj-semantic"]="face_retrieval functional_correspondence_blink functional_correspondence_funk_point handwritten_retrieval image2image_retrieval person_reid semantic_correspondence_blink semantic_correspondence_misc210k sketch2image_retrieval spot_the_similarity text2image_retrieval vehicle_retrieval visual_correspondence_blink visual_correspondence_scannet visually_grounded_reasoning_marvl"
domain_tasks["High-level-sub-semantic"]="casuality_reasoning_next_qa emotion_recognition_expw"
domain_tasks["Low-level-semantic"]="forensic_detection_blink forensic_detection_forgerynet visual_quality_assessment_ve_lol_l"

# MMMU
# domain_tasks["All"]="Accounting Agriculture Architecture_and_Engineering Art Art_Theory Basic_Medical_Science Biology Chemistry Clinical_Medicine Computer_Science Design Diagnostics_and_Laboratory_Medicine Economics Electronics Energy_and_Power Finance Geography History Literature Manage Marketing Materials Math Mechanical_Engineering Music Pharmacy Physics Psychology Public_Health Sociology"

# MMBench
# domain_tasks["finegrained_perception_cross_instance"]="spatial_relationship attribute_comparison action_recognition"
# domain_tasks["relation_reasoning"]="nature_relation physical_relation social_relation"
# domain_tasks["logic_reasoning"]="future_prediction structuralized_imagetext_understanding"
# domain_tasks["attribute_reasoning"]="physical_property_reasoning identity_reasoning function_reasoning"
# domain_tasks["coarse_perception"]="image_scene image_quality image_topic image_emotion image_style"
# domain_tasks["finegrained_perception_instance_level"]="object_localization attribute_recognition celebrity_recognition ocr"

# ScienceQA
# domain_tasks["All"]="All"

# prompting_methods=("Default" "Multi_Caption_Detail" "Multi_Caption" "Multi_Attention_Caption_Detail" "Multi_Duty_Decompose" "Multi_Scene_Graph" "Multi_Custom" "Multi_Custom4" "Multi_Attention_Caption" "Multi_Custom7" "Multi_Custom8" "Multi_Custom9" "Multi_Custom12")
# prompting_methods=("Default" "Single_Caption" "Single_Caption_Detail" "Single_Attention_Caption" "Single_Attention_Caption_Detail" "Single_Scene_Graph" "Single_Duty_Decompose" "Single_Custom" "Single_Custom2" "Single_Custom7")
prompting_methods=("Default" "Single_Caption" "Single_Caption_Detail" "Single_Attention_Caption" "Single_Attention_Caption_Detail" "Single_Scene_Graph" "Single_Duty_Decompose" "Single_Custom" "Single_Custom2" "Single_Custom7" "Multi_Caption" "Multi_Caption_Detail" "Multi_Attention_Caption" "Multi_Attention_Caption_Detail" "Multi_Scene_Graph" "Multi_Duty_Decompose" "Multi_Similar_Difference" "Multi_Custom" "Multi_Custom4" "Multi_Custom7")

window_name="${MODEL}_${DATASET}_new"

if ! tmux has-session -t $window_name 2>/dev/null; then
    tmux new-session -d -s $window_name
fi

DEVICES="1,2,0"

for DOMAIN in "${!domain_tasks[@]}"; do
    IFS=' ' read -r -a task_list <<< "${domain_tasks[$DOMAIN]}"
    # for llava-OV
    tmux send-keys -t $window_name:0 "conda activate llava" Enter
    # for others
    tmux send-keys -t $window_name:0 "conda activate /nfs/data/hgzhou42/shared_conda/environments/internvl/" Enter
    tmux send-keys -t $window_name:0 "export CUDA_VISIBLE_DEVICES=$DEVICES" Enter
    
    for METHOD in "${prompting_methods[@]}"; do
        for task in "${task_list[@]}"; do
            tmux send-keys -t $window_name:$index "python I2T_inference.py --dataset $DATASET --domain $DOMAIN --task $task --engine $MODEL --max-new-tokens 512 --prompting_method $METHOD" Enter
        done
    done
done
