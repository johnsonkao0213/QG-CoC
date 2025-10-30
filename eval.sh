#!/bin/bash

MODEL=$1
DATASET=$2
# METHOD=$3

# Declare an associative array to map domains to their task lists
declare -A domain_tasks

# Define task lists for each domain
# domain_tasks["2D-spatial"]="Homography_estimation Icon_Question_Answering_with_Spatial_Context Image_Captioning_with_Spatial_Context Image_Spatial_Transformation_Estimation jigsaw_puzzle_solving point_tracking ravens_progressive_matrices single_object_tracking "
# domain_tasks["3D-spatial"]="Egocentric_Video_QuestionAnswering Multiview_Action_Recognition Multiview_reasoning threed_cad_recognition threeD_Depth_Estimation threed_indoor_recognition threeD_Object_Detection threeD_Object_Tracking threeD_Pose_Estimation threeD_question_answering threeD_Scene_Reconstruction"
# domain_tasks["3D-spatial"]="Multiview_reasoning threed_cad_recognition threeD_Depth_Estimation threed_indoor_recognition threeD_Object_Detection threeD_Object_Tracking threeD_Pose_Estimation threeD_question_answering threeD_Scene_Reconstruction"
# domain_tasks["Continuous-temporal"]="action_quality_assessment general_action_recognition meme_vedio_understanding mevis next_img_prediction temporal_localization temporal_ordering video_captioning"
# domain_tasks["Discrete-temporal"]="gui_app_recognition gui_next_action_prediction textual_cloze visual_cloze visual_coherence visual_ordering"
# domain_tasks["High-level-obj-semantic"]="face_retrieval functional_correspondence_blink functional_correspondence_funk_point handwritten_retrieval image2image_retrieval person_reid semantic_correspondence_blink semantic_correspondence_misc210k sketch2image_retrieval spot_the_similarity text2image_retrieval vehicle_retrieval visual_correspondence_blink visual_correspondence_scannet visually_grounded_reasoning_marvl visually_grounded_reasoning_nlvr2"
# domain_tasks["High-level-sub-semantic"]="casuality_reasoning_next_qa casuality_reasoning_var emotion_recognition_expw emotion_recognition_findingemo multiple_image_captioning"
# domain_tasks["High-level-sub-semantic"]="casuality_reasoning_next_qa emotion_recognition_expw"
# domain_tasks["Low-level-semantic"]="forensic_detection_blink forensic_detection_forgerynet visual_quality_assessment_ve_lol_l"

# domain_tasks["All"]="EmojiAlgebra FuncRead GeomShape GeomCost Collisions Clocks Schedule Charts CodeEdit Isomorphism Maps RefCoco IQ"
domain_tasks["All"]="Geographic_Understanding Diagram_Understanding Image_Text_Matching Difference_Spotting Visual_Retrieval Counting Attribute_Similarity Scene_Understanding Action_Understanding Visual_Grounding Cartoon_Understanding Ordering"

# Loop over each domain and its tasks
for DOMAIN in "${!domain_tasks[@]}"; do
    window_name="${MODEL}_${DATASET}_${DOMAIN}_eval"
    tmux new-session -d -s $window_name
    index=0

    # Get the task list for the current domain
    IFS=' ' read -r -a task_list <<< "${domain_tasks[$DOMAIN]}"

    for task in "${task_list[@]}"; do
        # Create a new window for each task
        tmux new-window -t $window_name:$index
        tmux send-keys -t $window_name:$index "conda activate /nfs/data/hgzhou42/shared_conda/environments/internvl/" Enter
        tmux send-keys -t $window_name:$index "python I2T_evaluate.py --dataset $DATASET --domain $DOMAIN --task $task --engine $MODEL" Enter
        ((index++))
    done
done
