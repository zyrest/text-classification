tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='TextCNN/Predictions/Reshape_1' \
    /home/cying/project/thuCNN/model/best_validation.pb \
    /home/cying/project/thuCNN/model/web_model
