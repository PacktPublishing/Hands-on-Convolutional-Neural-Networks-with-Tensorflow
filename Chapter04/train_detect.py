def loss_layer(self, predicts, labels, scope='loss_layer'):
    with tf.variable_scope(scope):
        predict_classes = tf.reshape(predicts[:, :self.boundary1],
                                     [self.batch_size, self.cell_size, self.cell_size, self.num_class])
        predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],
                                    [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
        predict_boxes = tf.reshape(predicts[:, self.boundary2:],
                                   [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])
        boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
        classes = labels[:, :, :, 5:]

        offset = tf.constant(self.offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
        offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
        predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                       (predict_boxes[:, :, :, :, 1] + tf.transpose(offset,
                                                                                    (0, 2, 1, 3))) / self.cell_size,
                                       tf.square(predict_boxes[:, :, :, :, 2]),
                                       tf.square(predict_boxes[:, :, :, :, 3])])
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

        iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                               boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

        # class_loss
        class_delta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                    name='class_loss') * self.class_scale

        # object_loss
        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                     name='object_loss') * self.object_scale

        # noobject_loss
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                                       name='noobject_loss') * self.noobject_scale

        # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                    name='coord_loss') * self.coord_scale

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)