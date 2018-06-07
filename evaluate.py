from __future__ import print_function
import argparse
import scipy.misc as misc
import tensorflow as tf
import numpy as np
import os

from deeplab_resnet import CumulativeNets
import ImageRecords
import BatchDataset as BD
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/home/zhouzhilong/HED-BSDS'
DATA_LIST_PATH = '/home/zhouzhilong/HED-BSDS/test.lst'
IGNORE_LABEL = 2
NUM_CLASSES = 1
NUM_STEPS = 200 # Number of images in the validation set.
RESTORE_FROM = '/home/zhouzhilong/deeplab-coutour/snapshots/model.ckpt-bsds-6e-06-40000'
MODEL_NAME = 'bsds-mm2018'
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    images = tf.placeholder(tf.float32, [None, None, None, 3])
    # labels = tf.placeholder(tf.float32, [None, None, None, 1])
    # step_ph = tf.placeholder(dtype=tf.float32, shape=())

    test_records = ImageRecords.get_records_BSDS_test(DATA_DIRECTORY)
    test_dataset_reader = BD.BatchDatset("test", test_records)

    # Create network.
    net = CumulativeNets({'data': images}, is_training=False)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['refine5_sigmoid']
    logits_stage1 = tf.sigmoid(net.layers['refine1_out'])
    logits_stage2 = tf.sigmoid(net.layers['refine2_out'])
    logits_stage3 = tf.sigmoid(net.layers['refine3_out'])
    logits_stage4 = tf.sigmoid(net.layers['refine4_out'])
    logits_stage5 = tf.sigmoid(net.layers['refine5_out'])



    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)

    test_dataset_reader.reset_batch_offset()

    # Iterate over test steps.
    for step in range(args.num_steps):
        test_imgs = test_dataset_reader.next_image()
        test_imgs = np.expand_dims(test_imgs, axis=0)
        preds = sess.run(raw_output, feed_dict={images: test_imgs})
        print(preds.shape)
        if not os.path.isdir('./result/' + MODEL_NAME):
            os.mkdir('./result/' + MODEL_NAME)
            os.mkdir('./result/' + MODEL_NAME + '/final_out/')
            os.mkdir('./result/' + MODEL_NAME + '/stage1/')
            os.mkdir('./result/' + MODEL_NAME + '/stage2/')
            os.mkdir('./result/' + MODEL_NAME + '/stage3/')
            os.mkdir('./result/' + MODEL_NAME + '/stage4/')
        stage_1, stage_2, stage_3, stage_4 = sess.run([logits_stage1, logits_stage2, logits_stage3, logits_stage4], feed_dict={images:test_imgs})
        misc.imsave('./result/' + MODEL_NAME + '/final_out/'+ test_dataset_reader.files[step]["filename"], np.squeeze(preds*255, axis=(0,3)).astype(np.uint8))
        misc.imsave('./result/' + MODEL_NAME + '/stage1/' + test_dataset_reader.files[step]["filename"],
                    np.squeeze(stage_1 * 255, axis=(0, 3)).astype(np.uint8))
        misc.imsave('./result/' + MODEL_NAME + '/stage2/' + test_dataset_reader.files[step]["filename"],
                    np.squeeze(stage_2 * 255, axis=(0, 3)).astype(np.uint8))
        misc.imsave('./result/' + MODEL_NAME + '/stage3/' + test_dataset_reader.files[step]["filename"],
                    np.squeeze(stage_3 * 255, axis=(0, 3)).astype(np.uint8))
        misc.imsave('./result/' + MODEL_NAME + '/stage4/' + test_dataset_reader.files[step]["filename"],
                    np.squeeze(stage_4 * 255, axis=(0, 3)).astype(np.uint8))

    
if __name__ == '__main__':
    main()
