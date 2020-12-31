class MaskFeatHead(tf.keras.layers.Layer):

    def __init__(self,
                num_classes,
                in_channels,
                level_sizes,
                out_channels,
                is_training_bn,
                batch_size,
                strategy,
                start_level,
                end_level,
                name="mask_feat_head"):
        super().__init__(name=name)
        self.in_channels = in_channels
        self.level_sizes = level_sizes
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes
        self.is_training_bn = is_training_bn
        self.strategy = strategy

        self.all_level_convs = []
        for i in range(start_level, end_level + 1):
            level_convs = tf.keras.Sequential()
            if i == 0:
                input_shape = (batch_size, level_sizes[i][0], level_sizes[i][1], self.in_channels)
                level_convs.add(
                    tf.keras.layers.Conv2D(
                        self.out_channels,
                        3,
                        padding='same',
                        input_shape=input_shape
                    )
                )
                self.all_level_convs.append(level_convs)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i==3 else self.in_channels
                    input_shape = (batch_size, self.level_sizes[j][0], self.level_sizes[j][1], chn)
                    level_convs.add(
                        tf.keras.layers.Conv2D(
                            self.out_channels,
                            3,
                            padding='same',
                            input_shape=input_shape
                        )
                    )
                    level_convs.add(
                        tf.keras.layers.UpSampling2D(
                            size=(2, 2),
                            interpolation='bilinear'
                        )
                    )
                    continue

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
        assert len(inputs) == (self.end_level - self.start_level + 1)

        feature_add_all_level = self.all_level_convs[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                x_range = tf.linspace(-1, 1, input_feat.shape[-2])
                y_range = tf.linspace(-1, 1, input_feat.shape[-3])
                y, x = tf.meshgrid(y_range, x_range)
                y = tf.broadcast_to(y, [input_feat.shape[0], -1, -1, 1])
                x = tf.broadcast_to(x, [input_feat.shape[0], 1, -1, -1])
                coord_feat = tf.concat([x, y], -1)
                input_p = tf.concat([input_p, coord_feat], -1)
                
            feature_add_all_level += self.all_level_convs[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred



