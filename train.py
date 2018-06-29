
from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc as misc
import tensorflow as tf
import numpy as np
import ImageRecords
import BatchDataset as BD
from model import CumulativeNets
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './dataset/HED-BSDS'
DATA_LIST_PATH = './dataset/HED-BSDS/train_pair.lst'
BATCH_SIZE = 1
INPUT_SIZE = '256, 256'
LEARNING_RATE = 6e-6
NUM_CLASSES = 1
NUM_STEPS = 40001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'model.ckpt'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 4000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
MODEL_NAME = 'bsds-mm2018-threshold=0.2-channel=1'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt' + '-' + str(MODEL_NAME) + '-' + str(LEARNING_RATE)
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def sigmoid_cross_entropy_balanced(logits, label, name='cross_entropy_loss'):

    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / ((1 - beta))

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_sum(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)

def main():
    """Create the model and start the training."""
    args = get_arguments()
    images = tf.placeholder(tf.float32, [None, None, None, 3])
    labels = tf.placeholder(tf.float32, [None, None, None, 1])
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    tf.set_random_seed(args.random_seed)

    train_records = ImageRecords.get_records_BSDS_train(DATA_DIRECTORY)
    image_options = {'resize': True, 'resize_size': h}
    train_dataset_reader = BD.BatchDatset("train", train_records, image_options)

    # Create network.
    net = CumulativeNets({'data': images}, is_training=args.is_training)


    # Predictions.
    raw_output = net.layers['refine5_sigmoid']
    logits_stage1 = net.layers['refine1_out']
    logits_stage2 = net.layers['refine2_out']
    logits_stage3 = net.layers['refine3_out']
    logits_stage4 = net.layers['refine4_out']
    logits_stage5 = net.layers['refine5_out']
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name and 'up' not in v.name and 'dsn' not in v.name and 'concat' not in v.name\
                    and 'multi' not in v.name and 'rcu' not in v.name and 'refine' not in v.name and 'final' not in v.name]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0

    label_proc = labels / 255
    # Pixel-wise softmax loss.
    loss = sigmoid_cross_entropy_balanced(logits=logits_stage1, label=label_proc) + sigmoid_cross_entropy_balanced(logits=logits_stage4, label=label_proc)\
         + sigmoid_cross_entropy_balanced(logits=logits_stage2, label=label_proc) + sigmoid_cross_entropy_balanced(logits=logits_stage3, label=label_proc)\
         + sigmoid_cross_entropy_balanced(logits=logits_stage5, label=label_proc)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = loss + tf.add_n(l2_losses)




    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
   
    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    opt_conv = tf.train.AdamOptimizer(learning_rate)
    opt_fc_w = tf.train.AdamOptimizer(learning_rate * 10.0)
    opt_fc_b = tf.train.AdamOptimizer(learning_rate * 20.0)

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
    
    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    running_loss = 0
    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        np_images, np_labels, epochs = train_dataset_reader.next_batch(BATCH_SIZE)
        feed_dict = { step_ph : step , images: np_images, labels: np_labels}
        if step % args.save_pred_every == 0:
            loss_value, out, _ = sess.run([reduced_loss, raw_output,  train_op], feed_dict=feed_dict)

            save(saver, sess, args.snapshot_dir, step)
            running_loss += loss_value
        else:
            loss_value, out,_ = sess.run([reduced_loss, raw_output, train_op], feed_dict=feed_dict)
            running_loss += loss_value
        if step % 100 == 0 and step != 0:
            print("----------------------------------------------------------------------")
            print("Ava loss: {:.3f}".format(running_loss/100.0))
            print("----------------------------------------------------------------------")
            running_loss = 0
            if not os.path.isdir('./pred_test/' + MODEL_NAME):
                os.mkdir('./pred_test/' + MODEL_NAME)
            misc.imsave('./pred_test/' + MODEL_NAME + '/' +  str(step) + ".png", (np.squeeze(out[0], axis=2) * 255).astype(np.uint8))
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

if __name__ == '__main__':
    main()
