'''
The code is modified from Deeplab-tensorflow
'''
from kaffe.tensorflow import Network
import tensorflow as tf

class CumulativeNets(Network):
      def setup(self, is_training):
            '''Network definition.
            
            Args:
            is_training: whether to update the running mean and variance of the batch normalisation layer.
                        If the batch size is small, it is better to keep the running mean and variance of 
                        the-pretrained model frozen.
            '''
            # Resnet101
            (self.feed('data')
                  .conv(7, 7, 64, 1, 1, biased=False, relu=False, name='conv1')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
                  .max_pool(3, 3, 2, 2, name='pool1')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

            (self.feed('bn_conv1')
                  .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='conv1_dsn')
                  .upsample(self.layers['data'], name='conv1_up')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv1_up_dsn')
            )

            (self.feed('pool1')
                  .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
                  .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

            (self.feed('bn2a_branch1',
                        'bn2a_branch2c')
                  .add(name='res2a')
                  .relu(name='res2a_relu')
                  .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
                  .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))


            (self.feed('res2a_relu',
                        'bn2b_branch2c')
                  .add(name='res2b')
                  .relu(name='res2b_relu')
                  .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
                  .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

            (self.feed('res2b_relu',
                        'bn2c_branch2c')
                  .add(name='res2c')
                  .relu(name='res2c_relu')
                  .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

            (self.feed('res2c')
                  .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='res2c_dsn')
                  .deconv(4, 4, 1, 2, 2, 2, biased=False, relu=False, name='res2c_up')
                  .upsample(self.layers['data'], name='res2c_up')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_up_conv')
            )
           
            (self.feed('res2c_relu')
                  .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
                  .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
                  .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

            (self.feed('bn3a_branch1',
                        'bn3a_branch2c')
                  .add(name='res3a')
                  .relu(name='res3a_relu')
                  .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
                  .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
                  .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b1_branch2c'))

            (self.feed('res3a_relu',
                        'bn3b1_branch2c')
                  .add(name='res3b1')
                  .relu(name='res3b1_relu')
                  .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
                  .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
                  .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b2_branch2c'))

            (self.feed('res3b1_relu',
                        'bn3b2_branch2c')
                  .add(name='res3b2')
                  .relu(name='res3b2_relu')
                  .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
                  .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
                  .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b3_branch2c'))

            (self.feed('res3b2_relu',
                        'bn3b3_branch2c')
                  .add(name='res3b3')
                  .relu(name='res3b3_relu')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

            (self.feed('res3b3')
                  .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='res3b3_dsn')
                  .deconv(8, 8, 1, 4, 4, 4, biased=False, relu=False, name='res3b3_up')
                  .upsample(self.layers['data'], name='res3b3_up')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res3b3_up_conv')
            )
            


            (self.feed('res3b3_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

            (self.feed('bn4a_branch1',
                        'bn4a_branch2c')
                  .add(name='res4a')
                  .relu(name='res4a_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b1_branch2c'))

            (self.feed('res4a_relu',
                        'bn4b1_branch2c')
                  .add(name='res4b1')
                  .relu(name='res4b1_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b2_branch2c'))

            (self.feed('res4b1_relu',
                        'bn4b2_branch2c')
                  .add(name='res4b2')
                  .relu(name='res4b2_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b3_branch2c'))

            (self.feed('res4b2_relu',
                        'bn4b3_branch2c')
                  .add(name='res4b3')
                  .relu(name='res4b3_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b4_branch2c'))

            (self.feed('res4b3')
                  .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='res4b3_dsn')
                  .deconv(8, 8, 1, 4, 4, 4, biased=False, relu=False, name='res4b3_up')
                  .upsample(self.layers['data'], name='res4b3_up')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_up_conv')
            )

      

            (self.feed('res4b3_relu',
                        'bn4b4_branch2c')
                  .add(name='res4b4')
                  .relu(name='res4b4_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b5_branch2c'))

            (self.feed('res4b4_relu',
                        'bn4b5_branch2c')
                  .add(name='res4b5')
                  .relu(name='res4b5_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b6_branch2c'))

            (self.feed('res4b5_relu',
                        'bn4b6_branch2c')
                  .add(name='res4b6')
                  .relu(name='res4b6_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b7_branch2c'))

            (self.feed('res4b6_relu',
                        'bn4b7_branch2c')
                  .add(name='res4b7')
                  .relu(name='res4b7_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b8_branch2c'))

            (self.feed('res4b7_relu',
                        'bn4b8_branch2c')
                  .add(name='res4b8')
                  .relu(name='res4b8_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b9_branch2c'))

            (self.feed('res4b8_relu',
                        'bn4b9_branch2c')
                  .add(name='res4b9')
                  .relu(name='res4b9_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b10_branch2c'))

            (self.feed('res4b9_relu',
                        'bn4b10_branch2c')
                  .add(name='res4b10')
                  .relu(name='res4b10_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b11_branch2c'))

            (self.feed('res4b10_relu',
                        'bn4b11_branch2c')
                  .add(name='res4b11')
                  .relu(name='res4b11_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b12_branch2c'))

            (self.feed('res4b11_relu',
                        'bn4b12_branch2c')
                  .add(name='res4b12')
                  .relu(name='res4b12_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b13_branch2c'))

            (self.feed('res4b12_relu',
                        'bn4b13_branch2c')
                  .add(name='res4b13')
                  .relu(name='res4b13_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b14_branch2c'))

            (self.feed('res4b13_relu',
                        'bn4b14_branch2c')
                  .add(name='res4b14')
                  .relu(name='res4b14_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b15_branch2c'))

            (self.feed('res4b14_relu',
                        'bn4b15_branch2c')
                  .add(name='res4b15')
                  .relu(name='res4b15_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b16_branch2c'))

            (self.feed('res4b15_relu',
                        'bn4b16_branch2c')
                  .add(name='res4b16')
                  .relu(name='res4b16_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b17_branch2c'))

            (self.feed('res4b16_relu',
                        'bn4b17_branch2c')
                  .add(name='res4b17')
                  .relu(name='res4b17_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b18_branch2c'))

            (self.feed('res4b17_relu',
                        'bn4b18_branch2c')
                  .add(name='res4b18')
                  .relu(name='res4b18_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b19_branch2c'))

            (self.feed('res4b18_relu',
                        'bn4b19_branch2c')
                  .add(name='res4b19')
                  .relu(name='res4b19_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b20_branch2c'))

            (self.feed('res4b19_relu',
                        'bn4b20_branch2c')
                  .add(name='res4b20')
                  .relu(name='res4b20_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b21_branch2c'))

            (self.feed('res4b20_relu',
                        'bn4b21_branch2c')
                  .add(name='res4b21')
                  .relu(name='res4b21_relu')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
                  .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
                  .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b22_branch2c'))

            (self.feed('res4b21_relu',
                        'bn4b22_branch2c')
                  .add(name='res4b22')
                  .relu(name='res4b22_relu')
                  .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

            (self.feed('res4b22_relu')
                  .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
                  .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
                  .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

            (self.feed('bn5a_branch1',
                        'bn5a_branch2c')
                  .add(name='res5a')
                  .relu(name='res5a_relu')
                  .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
                  .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
                  .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

            (self.feed('res5a_relu',
                        'bn5b_branch2c')
                  .add(name='res5b')
                  .relu(name='res5b_relu')
                  .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
                  .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
                  .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
                  .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
                  .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
                  .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

            (self.feed('res5b_relu',
                        'bn5c_branch2c')
                  .add(name='res5c')
                  .relu(name='res5c_relu')
                  .atrous_conv(3, 3, 1, 6, padding='SAME', relu=False, name='fc1_voc12_c0'))

            (self.feed('res5c')
                  .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='res5c_dsn')
                  .deconv(8, 8, 1, 4, 4, 4, biased=False, relu=False, name='res5c_up')
                  .upsample(self.layers['data'], name='res5c_up')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res5c_up_conv')
            )
            
            (self.feed('res5c_relu')
                  .atrous_conv(3, 3, 1, 12, padding='SAME', relu=False, name='fc1_voc12_c1'))

            (self.feed('res5c_relu')
                  .atrous_conv(3, 3, 1, 18, padding='SAME', relu=False, name='fc1_voc12_c2'))

            (self.feed('res5c_relu')
                  .atrous_conv(3, 3, 1, 24, padding='SAME', relu=False, name='fc1_voc12_c3'))
            
            (self.feed('fc1_voc12_c0',
                        'fc1_voc12_c1',
                        'fc1_voc12_c2',
                        'fc1_voc12_c3',)
                  .add(name='fc_voc12')
                  .deconv(8, 8, 1, 4, 4, 4, biased=False, relu=False, name='fc1_voc12')
                  .upsample(self.layers['data'], name='fc1_voc12_up')
                  .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='fc1_voc12_up_conv')
                  )
            


            # CRA 1
            # SRU_res2c
            (self.feed('res2c_up')
                  .relu(name='rcu1_res2c_relu1')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_res2c_conv1')
                  .relu(name='rcu1_res2c_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_res2c_conv2')
            )

            (self.feed('res2c_up_conv',
                        'rcu1_res2c_conv2')
                  .add(name='rcu1_res2c'))

            (self.feed('rcu1_res2c')
                  .relu(name='rcu2_res2c_relu1')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_res2c_conv1')
                  .relu(name='rcu2_res2c_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_res2c_conv2'))

            (self.feed('rcu1_res2c',
                        'rcu2_res2c_conv2')
                  .add(name='rcu2_res2c')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='multi_fusion_res2c'))

            # SRU_conv1_up
            (self.feed('conv1_up')
            .relu(name='rcu1_conv1_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_conv1_conv1')
            .relu(name='rcu1_conv1_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_conv1_conv2')
            )

            (self.feed('conv1_up_dsn',
                        'rcu1_conv1_conv2')
            .add(name='rcu1_conv1'))


            (self.feed('rcu1_conv1')
            .relu(name='rcu2_conv1_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_conv1_conv1')
            .relu(name='rcu2_conv1_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_conv1_conv2'))

            (self.feed('rcu1_conv1',
                        'rcu2_conv1_conv2')
                  .add(name='rcu2_conv1')
                  .conv(3, 3, 256, 1 ,1, biased=False, relu=False, name='multi_fusion_conv1'))

            # Multi-Fusion & Attention
            (self.feed('multi_fusion_res2c',
                        'multi_fusion_conv1')
                  .add(name='multi_fusion1_out')
                  .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='multi_fusion1_out2')
                  .softmax(name='multi_fusion1_softmax')
                  .multiply(self.layers['multi_fusion1_out'], name='multi_fusion1_attention')
                  .conv(1, 1, 21, 1, 1, biased=False, relu=False, name='refine1_conv1')
                  .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='refine1_out'))
            

            # CRA 2
            # SRU1_refine1
            (self.feed('refine1_out')
                  .relu(name='rcu1_refine1_relu1')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_refine1_conv1')
                  .relu(name='rcu1_refine1_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_refine1_conv2')
            )

            (self.feed('multi_fusion1_attention',
                        'rcu1_refine1_conv2')
                  .add(name='rcu1_refine1'))

            (self.feed('rcu1_refine1')
                  .relu(name='rcu2_refine1_relu1')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_refine1_conv1')
                  .relu(name='rcu2_refine1_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_refine1_conv2'))

            (self.feed('rcu1_refine1',
                        'rcu2_refine1_conv2')
                  .add(name='rcu2_refine1')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='multi_fusion_refine1'))

            # SRU_res3b3
            (self.feed('res3b3_up')
            .relu(name='rcu1_res3b3_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_res3b3_conv1')
            .relu(name='rcu1_res3b3_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_res3b3_conv2')
            )

            (self.feed('res3b3_up_conv',
                        'rcu1_res3b3_conv2')
            .add(name='rcu1_res3b3'))

            (self.feed('rcu1_res3b3')
            .relu(name='rcu2_res3b3_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_res3b3_conv1')
            .relu(name='rcu2_res3b3_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_res3b3_conv2'))

            (self.feed('rcu1_res3b3',
                        'rcu2_res3b3_conv2')
                  .add(name='rcu2_res3b3')
                  .conv(3, 3, 256, 1 ,1, biased=False, relu=False, name='multi_fusion_res3b3'))

            # Multi-Fusion &  Attention
            (self.feed('multi_fusion_refine1',
                        'multi_fusion_res3b3')
                  .add(name='multi_fusion2_out')
                  .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='multi_fusion2_out2')
                  .softmax(name='multi_fusion2_softmax')
                  .multiply(self.layers['multi_fusion2_out'], name='multi_fusion2_attention')
                  .conv(1, 1, 21, 1, 1, biased=False, relu=False, name='refine2_conv1')
                  .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='refine2_out'))
            

            # CRA 3
            # SRU_refine2
            (self.feed('refine2_out')
                  .relu(name='rcu1_refine2_relu1')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_refine2_conv1')
                  .relu(name='rcu1_refine2_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_refine2_conv2')
            )

            (self.feed('multi_fusion2_attention',
                        'rcu1_refine2_conv2')
                  .add(name='rcu1_refine2'))


            (self.feed('rcu1_refine2')
                  .relu(name='rcu2_refine2_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_refine2_conv1')
                  .relu(name='rcu2_refine2_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_refine2_conv2'))

            (self.feed('rcu1_refine2',
                        'rcu2_refine2_conv2')
                  .add(name='rcu2_refine2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='multi_fusion_refine2'))

            # SRU_res4b3
            (self.feed('res4b3_up')
                  .relu(name='rcu1_res4b3_relu1')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_res4b3_conv1')
                  .relu(name='rcu1_res4b3_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_res4b3_conv2')
            )

            (self.feed('res4b3_up_conv',
                        'rcu1_res4b3_conv2')
            .add(name='rcu1_res4b3'))

            (self.feed('rcu1_res4b3')
                  .relu(name='rcu2_res4b3_relu1')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_res4b3_conv1')
                  .relu(name='rcu2_res4b3_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_res4b3_conv2'))

            (self.feed('rcu1_res4b3',
                        'rcu2_res4b3_conv2')
                  .add(name='rcu2_res4b3')
                  .conv(3, 3, 256, 1 ,1, biased=False, relu=False, name='multi_fusion_res4b3'))

            # Multi-Fusion & Attention
            (self.feed('multi_fusion_refine2',
                        'multi_fusion_res4b3')
                  .add(name='multi_fusion3_out')
                  .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='multi_fusion3_out2')
                  .softmax(name='multi_fusion3_softmax')
                  .multiply(self.layers['multi_fusion3_out'], name='multi_fusion3_attention')
                  .conv(1, 1, 21, 1, 1, biased=False, relu=False, name='refine3_conv1')
                  .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='refine3_out'))
            


            # CRA 4
            # SRU_refine3
            (self.feed('refine3_out')
            .relu(name='rcu1_refine3_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_refine3_conv1')
            .relu(name='rcu1_refine3_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_refine3_conv2')
            )

            (self.feed('multi_fusion3_attention',
                        'rcu1_refine3_conv2')
            .add(name='rcu1_refine3'))

            (self.feed('rcu1_refine3')
            .relu(name='rcu2_refine3_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_refine3_conv1')
            .relu(name='rcu2_refine3_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_refine3_conv2'))

            (self.feed('rcu1_refine3',
                        'rcu2_refine3_conv2')
            .add(name='rcu2_refine3')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='multi_fusion_refine3'))

            # SRU_res5c
            (self.feed('res5c_up')
            .relu(name='rcu1_res5c_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_res5c_conv1')
            .relu(name='rcu1_res5c_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_res5c_conv2')
            )

            (self.feed('res5c_up_conv',
                        'rcu1_res5c_conv2')
            .add(name='rcu1_res5c'))

            (self.feed('rcu1_res5c')
                  .relu(name='rcu2_res5c_relu1')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_res5c_conv1')
                  .relu(name='rcu2_res5c_relu2')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_res5c_conv2'))

            (self.feed('rcu1_res5c',
                        'rcu2_res5c_conv2')
                  .add(name='rcu2_res5c')
                  .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='multi_fusion_res5c'))

            # Multi-Fusion & Attention
            (self.feed('multi_fusion_refine3',
                        'multi_fusion_res5c')
                  .add(name='multi_fusion4_out')
                  .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='multi_fusion1_out4')
                  .softmax(name='multi_fusion4_softmax')
                  .multiply(self.layers['multi_fusion4_out'], name='multi_fusion4_attention')
                  .conv(1, 1, 21, 1, 1, biased=False, relu=False, name='refine4_conv1')
                  .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='refine4_out'))
            
            # CRA 5
            # SRU_refine4
            (self.feed('refine4_out')
            .relu(name='rcu1_refine4_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_refine4_conv1')
            .relu(name='rcu1_refine4_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_refine4_conv2')
            )

            (self.feed('multi_fusion4_out',
                        'rcu1_refine4_conv2')
            .add(name='rcu1_refine4'))

            (self.feed('rcu1_refine4')
            .relu(name='rcu2_refine4_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_refine4_conv1')
            .relu(name='rcu2_refine4_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_refine4_conv2'))

            (self.feed('rcu1_refine4',
                        'rcu2_refine4_conv2')
            .add(name='rcu2_refine4')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='multi_fusion_refine4'))

            # SRU_fc1_voc12_up
            (self.feed('fc1_voc12_up')
            .relu(name='rcu1_fc1_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_fc1_conv1')
            .relu(name='rcu1_fc1_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu1_fc1_conv2')
            )

            (self.feed('fc1_voc12_up_conv',
                        'rcu1_fc1_conv2')
            .add(name='rcu1_fc1'))


            (self.feed('rcu1_fc1')
            .relu(name='rcu2_fc1_relu1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_fc1_conv1')
            .relu(name='rcu2_fc1_relu2')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='rcu2_fc1_conv2'))

            (self.feed('rcu1_fc1',
                        'rcu2_fc1_conv2')
            .add(name='rcu2_fc1')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='multi_fusion_fc1'))

            # Multi-Fusion & Attention
            (self.feed('multi_fusion_refine4',
                        'multi_fusion_fc1')
            .add(name='multi_fusion5_out')
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='multi_fusion5_out2')
            .softmax(name='multi_fusion5_softmax')
            .multiply(self.layers['multi_fusion5_out'], name='multi_fusion5_attention')
            .conv(1, 1, 21, 1, 1, biased=False, relu=False, name='refine5_conv1')
            .conv(1, 1, 1, 1, 1, biased=False, relu=False, name='refine5_out')
            .sigmoid(name='refine5_sigmoid'))



