def read_setting():
    """ return nework_setting dictionary which have following keys
    "learning_rate" : learning rate when weight update 
    "batch_size" : how often weights are updated
    "layer_settings" : list of setting details for each layer.
    """
    network_setting = {
        "learning_rate" : 0.01,
        "batch_size" : 10,
        "layers_setting" : [
            {"layer_type" : "ConvolutionalLayer", 
             "setting_detail" : {"input_kernel_size" : 1,
                                 "input_shape" : (28, 28),
                                 "output_kernel_size" : 10,
                                 "output_shape" : (24, 24),
                                 "window_size" : 5,
                                 "step_size" : 1
                                 }
             },
            
            {"layer_type" : "MaxPooling", "setting_detail" : 0},
            
            {"layer_type" : "Activation", 
             "setting_detail" : { "activation_type" :"ReLU"}
             },
            
            {"layer_type" : "ConvolutionalLayer",
             "setting_detail" : {"input_kernel_size" : 10,
                                 "input_shape" : (12, 12),
                                 "output_kernel_size" : 12,
                                 "output_shape" : (10, 10),
                                 "window_size" : 3,
                                 "step_size" : 1
                                 }
             },
            {"layer_type" : "MaxPooling", "setting_detail" : 0},

            {"layer_type" : "Activation", 
             "setting_detail" : { "activation_type" :"ReLU"}
             },

            {"layer_type" : "FCLayer", 
             "setting_detail" : { "input_num" : (12, 5, 5),
                                  "output_num" : 128,
                                  }
             },
            
            {"layer_type" : "Activation", 
             "setting_detail" : { "activation_type" :"tanh"}
             },
            
            {"layer_type" : "FCLayer", 
             "setting_detail" : { "input_num" : 128,
                                  "output_num" : 10,
                                  }
             },
            
            {"layer_type" : "Activation", 
             "setting_detail" : { "activation_type" :"softmax"}
             }
            ]
        }
    return network_setting
