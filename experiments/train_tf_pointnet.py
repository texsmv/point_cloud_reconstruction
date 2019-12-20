
import tensorflow as tf
import os
import logging
import sys
from models.model_emd_pointNet import get_loss, get_model, placeholder_inputs
from datasets.tf_VesselsPoint import generator, get_number_pc
from tensorflow.data import Dataset
from utils.pcutil import show_pc
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG,filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


# tf.logging.set_verbosity(tf.logging.WARN)


BATCH_SIZE = 5
NUM_POINT = 2048
MAX_EPOCH = 40


is_training_pl = True

bn_decay = True

n_pc = get_number_pc()

tf.reset_default_graph()

# np.random.seed(42)
tf.set_random_seed(2019)
 

with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            dataset = Dataset.from_generator(generator, (tf.float32), output_shapes=(tf.TensorShape([ NUM_POINT, 3])), args=([BATCH_SIZE * int(n_pc/BATCH_SIZE)]))
            # dataset = Dataset.from_generator(generator, (tf.float32, tf.float32), output_shapes=(tf.TensorShape([1000]), tf.TensorShape([ NUM_POINT, 3])))
            # dataset = dataset.repeat(1)
            dataset = dataset.shuffle(100)
            dataset = dataset.batch(BATCH_SIZE)

            # iterator = dataset.make_one_shot_iterator()
            iterator = dataset.make_initializable_iterator()


            # features, pointclouds_pl = iterator.get_next()0
            pointclouds_pl = iterator.get_next()
            pointclouds_pl = tf.reshape(pointclouds_pl, (BATCH_SIZE, NUM_POINT, 3))
            print("---------------------->   ",pointclouds_pl)
            # print(batch)




            # pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)


            pred, end_points = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, num_points=NUM_POINT, batch_size=BATCH_SIZE)
            loss, end_points = get_loss(pred, pointclouds_pl, end_points)

            optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
            train_op = optimizer.minimize(loss)


            saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)


            ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'end_points': end_points}

            init = tf.global_variables_initializer()
            sess.run(init)

            if os.path.exists("results/tf_pointNet/model_final.meta"):
                print("loading")
                imported_meta = tf.train.import_meta_graph("results/tf_pointNet/model_final.meta")
                imported_meta.restore(sess, tf.train.latest_checkpoint('results/tf_pointNet/'))


            loss_sum = 0
            pcloss_sum = 0
            batch_id = 0
            for epoch in range(MAX_EPOCH):
                sess.run(iterator.initializer)
                
                try:
                    while(True):
                        
                        # print("epoch")
                        pc_orig, _, loss_val, pcloss_val, pred_val = sess.run([ops['pointclouds_pl'], ops['train_op'], ops['loss'], ops['end_points']['pcloss'], ops['pred']])
                        sess.run(train_op)

                        loss_sum += loss_val
                        pcloss_sum += pcloss_val
                        batch_id += 1
                        if batch_id == 30 and (epoch % 1 == 0):
                            pc_pred = pred_val #sess.run(pred)
                            pc_orig = pc_orig #sess.run(pointclouds_pl)
                            
                            for i in range(5):
                                show_pc(pc_orig[i])
                                plt.savefig('results/tf_pointNet/plots/'+str(epoch)+'_'+str(i)+'_real'+'.png')
                                show_pc(pc_pred[i])
                                plt.savefig('results/tf_pointNet/plots/'+str(epoch)+'_'+str(i)+'_reconstructed'+'.png')
                                # plt.show()
                                # plt.cla()
                                # plt.cl() 

                                        


                except tf.errors.OutOfRangeError:
                    pass

                
                logging.debug("epoch: "+str(epoch)+" :"+ " loss: "+str(loss_sum/n_pc) + "   pcloss: " +str(pcloss_sum/n_pc))
                loss_sum = 0
                pcloss_sum = 0
                batch_id = 0

                saver.save(sess, 'results/tf_pointNet/model_iter.ckpt', global_step=epoch)
            
                # feat, poi = batch.eval(session=sess)

                # print(feat.shape)
                # print("----------->",batch.shape)

            saver.save(sess, 'results/tf_pointNet/model_final')

            






