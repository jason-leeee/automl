import tensorflow as tf
import numpy as np

class MaskFeatHead(tf.keras.layers.Layer):

    def __init__(self,
                num_classes,
                out_channels,
                is_training_bn,
                strategy,
                start_level,
                end_level,
                name="mask_feat_head",
                **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes
        self.is_training_bn = is_training_bn
        self.strategy = strategy

        self.all_level_convs = []
        for i in range(start_level, end_level + 1):
            assert start_level >= 2, "minimum start level should be at least 2!"
            level_convs = tf.keras.Sequential()
            if i == 2:
                
                level_convs.add(
                    tf.keras.layers.Conv2D(
                        self.out_channels,
                        3,
                        padding='same'
                    )
                )
                self.all_level_convs.append(level_convs)
                continue

            for j in range(i-2):
                level_convs.add(
                    tf.keras.layers.Conv2D(
                        self.out_channels,
                        3,
                        padding='same'                   
                    )
                )
                level_convs.add(
                    tf.keras.layers.UpSampling2D(
                        size=(2, 2),
                        interpolation='bilinear'
                    )
                )
            self.all_level_convs.append(level_convs)
        
        self.conv_pred = tf.keras.layers.Conv2D(self.num_classes, 1, padding='valid')

    def call(self, inputs):
        print(inputs)
        assert len(inputs) == (self.end_level - self.start_level + 1)
        #for input in inputs:
        #    print("fpn feature map size:", input.shape)

        feature_add_all_level = self.all_level_convs[0](inputs[0])

        for i in range(1, len(inputs)):
            input_p = inputs[i]
            print(input_p)
            print('input shape is:', input_p.shape)
            if i == 3:
                input_feat = input_p
                x_range = tf.linspace(-1, 1, input_feat.shape[-3])
                y_range = tf.linspace(-1, 1, input_feat.shape[-2])
                y, x = tf.meshgrid(y_range, x_range)  
                y = tf.cast(tf.expand_dims(y, -1), input_feat.dtype)
                x = tf.cast(tf.expand_dims(x, -1), input_feat.dtype)
                print(y) 
                y = tf.broadcast_to(y, [input_feat.shape[0], -1, -1, 1])
                x = tf.broadcast_to(x, [input_feat.shape[0], -1, -1, 1])
                coord_feat = tf.concat([x, y], -1)
                input_p = tf.concat([input_p, coord_feat], -1)
                
            feature_add_all_level += self.all_level_convs[i](input_p)
        #for layer in self.all_level_convs:
        #    layer.summary()
        feature_pred = self.conv_pred(feature_add_all_level)
        #print("Final feature map shape:", feature_pred.shape)
        return feature_pred



