import tensorflow as tf
import numpy as np
import cv2

class HED:

    def __init__(self, config):
        self.config = config
        self.path_npz = config.path_npz

    def load_npz(self):
        obj = np.load(self.path_npz)
        obj = dict(obj)       #key:层名, value:参数值
        return obj

    def conv2d(self, input, filter, name):
        return tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding= 'SAME', name=name)

    def BilinearUpSample(self, x, shape):
        """
        Deterministic bilinearly-upsample the input images.

        Args:
            x (tf.Tensor): a NHWC tensor
            shape (int): the upsample factor

        Returns:
            tf.Tensor: a NHWC tensor.
        """
        inp_shape = x.shape.as_list()
        ch = inp_shape[3]
        assert ch is not None

        shape = int(shape)
        filter_shape = 2 * shape

        def bilinear_conv_filler(s):
            """
            s: width, height of the conv filter
            See https://github.com/BVLC/caffe/blob/master/include%2Fcaffe%2Ffiller.hpp#L244
            """
            f = np.ceil(float(s) / 2)
            c = float(2 * f - 1 - f % 2) / (2 * f)
            ret = np.zeros((s, s), dtype='float32')
            for x_s in range(s):
                for y in range(s):
                    ret[x_s, y] = (1 - abs(x_s / f - c)) * (1 - abs(y / f - c))
            return ret

        w = bilinear_conv_filler(filter_shape)
        w = np.repeat(w, ch * ch).reshape((filter_shape, filter_shape, ch, ch))

        weight_var = tf.constant(w, tf.float32,
                                 shape=(filter_shape, filter_shape, ch, ch),
                                 name='bilinear_upsample_filter')
        x = tf.pad(x, [[0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1], [0, 0]], mode='SYMMETRIC')
        out_shape = tf.shape(x) * tf.constant([1, shape, shape, 1], tf.int32)
        deconv = tf.nn.conv2d_transpose(x, weight_var, out_shape,
                                        [1, shape, shape, 1], 'SAME')
        edge = shape * (shape - 1)
        deconv = deconv[:, edge:-edge, edge:-edge, :]

        if inp_shape[1]:
            inp_shape[1] *= shape
        if inp_shape[2]:
            inp_shape[2] *= shape
        deconv.set_shape(inp_shape)
        return deconv

    def branch(self, name, filters, bias, l, up):
        with tf.variable_scope(name):
            l = tf.nn.conv2d(l,filters,strides=[1,1,1,1],padding='SAME')
            l = tf.identity(l + bias)
            while up != 1:
                l = self.BilinearUpSample(l, 2)
                up = up / 2
            return l

    def maxPooling(self, input, pool_size, name):
        ''' pool_size:kernel size'''
        return tf.nn.max_pool(input, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME', name=name)

    def testing_frame(self, image):
        image = image - tf.constant([104, 116, 122], dtype='float32')
        obj = self.load_npz()
        with tf.name_scope('part1'):
            l = tf.nn.relu(self.conv2d(image,obj['conv1_1/W'],'conv1_1')+obj['conv1_1/b'])
            l = tf.nn.relu(self.conv2d(l,obj['conv1_2/W'],'conv1_2')+obj['conv1_2/b'])
            b1 = self.branch('branch1',obj['branch1/convfc/W'],obj['branch1/convfc/b'],l,1)
            l = self.maxPooling(l,2,'pool1')

        with tf.name_scope('part2'):
            l = tf.nn.relu(self.conv2d(l, obj['conv2_1/W'], 'conv2_1') + obj['conv2_1/b'])
            l = tf.nn.relu(self.conv2d(l, obj['conv2_2/W'], 'conv2_2') + obj['conv2_2/b'])
            b2 = self.branch('branch2',obj['branch2/convfc/W'],obj['branch2/convfc/b'],l,2)
            l = self.maxPooling(l,2,'pool2')

        with tf.name_scope('part3'):
            l = tf.nn.relu(self.conv2d(l, obj['conv3_1/W'], 'conv3_1') + obj['conv3_1/b'])
            l = tf.nn.relu(self.conv2d(l, obj['conv3_2/W'], 'conv3_2') + obj['conv3_2/b'])
            l = tf.nn.relu(self.conv2d(l, obj['conv3_3/W'], 'conv3_3') + obj['conv3_3/b'])
            b3 = self.branch('branch3',obj['branch3/convfc/W'],obj['branch3/convfc/b'],l,4)
            l = self.maxPooling(l, 2, 'pool3')

        with tf.name_scope('part4'):
            l = tf.nn.relu(self.conv2d(l, obj['conv4_1/W'], 'conv4_1') + obj['conv4_1/b'])
            l = tf.nn.relu(self.conv2d(l, obj['conv4_2/W'], 'conv4_2') + obj['conv4_2/b'])
            l = tf.nn.relu(self.conv2d(l, obj['conv4_3/W'], 'conv4_3') + obj['conv4_3/b'])
            b5 = self.branch('branch4',obj['branch4/convfc/W'],obj['branch4/convfc/b'],l,8)
            l = self.maxPooling(l, 2, 'pool4')

        with tf.name_scope('part5'):
            l = tf.nn.relu(self.conv2d(l, obj['conv5_1/W'], 'conv5_1') + obj['conv5_1/b'])
            l = tf.nn.relu(self.conv2d(l, obj['conv5_2/W'], 'conv5_2') + obj['conv5_2/b'])
            l = tf.nn.relu(self.conv2d(l, obj['conv5_3/W'], 'conv5_3') + obj['conv5_3/b'])
            b4 = self.branch('branch5',obj['branch5/convfc/W'],obj['branch5/convfc/b'],l,16)

        with tf.name_scope('final'):
            final_map = self.conv2d(tf.concat([b1,b2,b3,b4,b5],3),obj['convfcweight/W'],name='convfcweight')
            final_map = tf.identity(final_map)
            image_list = []
            for idx, b in enumerate([b1, b2, b3, b4, b5, final_map]):
                output = tf.nn.sigmoid(b, name='output{}'.format(idx + 1))
                image_list.append(output)
        return image_list

    def image_save(self, image_list): # [1,img_width,img_height,1]
        sess =  tf.Session()
        for i in range(len(image_list)):
            temp = sess.run(image_list[i])
            output_path = self.config.output_dir + str(i) + '.jpg'
            cv2.imwrite(output_path, 255 - temp[0] * 255)
            print(i)
        sess.close()

    def predict(self, image_path):
        im = cv2.imread(image_path)
        assert im is not None
        im = cv2.resize(im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16))[None, :, :, :].astype('float32')
        image_list = self.testing_frame(im)
        self.image_save(image_list)
