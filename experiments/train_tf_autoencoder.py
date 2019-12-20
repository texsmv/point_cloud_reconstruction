
import tensorflow as tf
import os
import logging
import sys
from models.model_emd import get_loss, get_model, placeholder_inputs
from datasets.tf_Vessels import generator, get_number_pc
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


EXPERIMENT_NAME = "tf_basis_autoencoder"


BATCH_SIZE = 5
NUM_POINT = 1024
MAX_EPOCH = 80
N_BASIS = 1024


is_training_pl = True
bn_decay = True

n_pc = get_number_pc()

tf.reset_default_graph()
# np.random.seed(42)
tf.set_random_seed(2019)
 

print(n_pc, "<_------------------------------------")

with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            shapes = ((N_BASIS), (NUM_POINT,3))
        
            dataset = Dataset.from_generator(generator, (tf.float32, tf.float32), output_shapes = shapes, args=([BATCH_SIZE * int(n_pc/BATCH_SIZE)]))

            # output_shapes=([tf.TensorShape([ N_BASIS]), tf.TensorShape([ NUM_POINT, 3]) ])
            # dataset = Dataset.from_generator(generator, (tf.float32, tf.float32), output_shapes=(tf.TensorShape([1000]), tf.TensorShape([ NUM_POINT, 3])))
            # dataset = dataset.repeat(1)
            dataset = dataset.shuffle(100)
            dataset = dataset.batch(BATCH_SIZE)

            # iterator = dataset.make_one_shot_iterator()
            iterator = dataset.make_initializable_iterator()

            pointcloud = tf.placeholder(tf.float32, shape=(None, NUM_POINT, 3))

            # features, basisPset_feature = iterator.get_next()0
            basisPset_feature, pointcloud = iterator.get_next()
            

            pointcloud = tf.reshape(pointcloud, (BATCH_SIZE, NUM_POINT, 3))
            basisPset_feature = tf.reshape(basisPset_feature, (BATCH_SIZE, N_BASIS))




            # basisPset_feature, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)


            pred, end_points = get_model(basisPset_feature, is_training_pl, bn_decay=bn_decay, batch_size=BATCH_SIZE, num_points=1024)
            loss, end_points = get_loss(pred, pointcloud, end_points)

            optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
            train_op = optimizer.minimize(loss)


            saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)


            ops = {'basisPset_feature': basisPset_feature,
               'labels_pl': basisPset_feature,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'end_points': end_points,
               'pointcloud': pointcloud}

            init = tf.global_variables_initializer()
            sess.run(init)

            if os.path.exists("results/" + EXPERIMENT_NAME + "/model_final.meta"):
                print("loading")
                imported_meta = tf.train.import_meta_graph("results/" + EXPERIMENT_NAME + "/model_final.meta")
                imported_meta.restore(sess, tf.train.latest_checkpoint('results/' + EXPERIMENT_NAME + '/'))


            loss_sum = 0
            pcloss_sum = 0
            batch_id = 0
            for epoch in range(MAX_EPOCH):
                sess.run(iterator.initializer)
                
                try:
                    while(True):
                        
                        print("-")
                        bps_feat, _, loss_val, pcloss_val, pred_val, pointcloud = sess.run([ops['basisPset_feature'], ops['train_op'], ops['loss'], ops['end_points']['pcloss'], ops['pred'], ops['pointcloud']])
                        sess.run(train_op)

                        loss_sum += loss_val
                        pcloss_sum += pcloss_val
                        batch_id += 1
                        if batch_id == 30 and (epoch % 1 == 0):
                            pc_pred = pred_val #sess.run(pred)
                            bps_feat = bps_feat #sess.run(basisPset_feature)
                            
                            for i in range(5):
                                show_pc(pc_pred[i])
                                plt.savefig('results/' + EXPERIMENT_NAME + '/plots/'+str(epoch)+'_'+str(i)+'_reconstructed'+'.png')
                                # plt.close()
                                show_pc(pointcloud[i])
                                plt.savefig('results/' + EXPERIMENT_NAME + '/plots/'+str(epoch)+'_'+str(i)+'_real'+'.png')
                                # plt.close()
                                

                                # plt.show()
                                # plt.cla()
                                # plt.cl() 

                                        


                except tf.errors.OutOfRangeError:
                    pass


                
                logging.debug("epoch: "+str(epoch)+" :"+ " loss: "+str(loss_sum/n_pc) + "   pcloss: " +str(pcloss_sum/n_pc))
                loss_sum = 0
                pcloss_sum = 0
                batch_id = 0

                saver.save(sess, 'results/' + EXPERIMENT_NAME + '/model_iter.ckpt', global_step=epoch)
            
                # feat, poi = batch.eval(session=sess)

                # print(feat.shape)
                # print("----------->",batch.shape)

            saver.save(sess, 'results/' + EXPERIMENT_NAME + '/model_final')

            






