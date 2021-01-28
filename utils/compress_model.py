import tensorflow as tf
import os

id = 66
path = '/home/liheng/nas/res/_soccer5/%i/' % id

ckpt = tf.train.get_checkpoint_state(path)

comm = 'cd %s && ' % path
# model_files = list(ckpt.all_model_checkpoint_paths) + [ckpt.model_checkpoint_path, ]
# comm = 'zip %i.zip %scheckpoint ' % (id, path) + '* '.join(model_files)
model_file = os.path.basename(ckpt.model_checkpoint_path)
comm += 'zip %i.zip checkpoint %s* graph.pbtxt events.out* && ' % (id, model_file)
comm += 'mv %i.zip ~/ && cd' % id
print(comm)
os.system(comm)