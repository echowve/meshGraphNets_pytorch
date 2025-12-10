# -*- encoding: utf-8 -*-
'''
@File    :   parse_tfrecord.py
@Author  :   jianglx 
@Version :   2.0
@Contact :   jianglx@whu.edu.cn
'''
import tensorflow as tf
import functools
import json
import os
import numpy as np
from packaging import version


if version.parse(tf.__version__) >= version.parse("1.15"):
    raise RuntimeError(
        f"当前 TensorFlow 版本为 {tf.__version__}，但本项目要求 tensorflow<1.15。"
        "请在其他环境安装TensorFlow：pip install 'tensorflow<1.15'"
    )

def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
      meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=1)
    ds = ds.prefetch(1)
    return ds


if __name__ == '__main__':
    
    tf_datasetPath = 'data'

    tf.enable_resource_variables() # type: ignore
    tf.enable_eager_execution() # type: ignore

    for split in ['train', 'test', 'valid']:
        ds = load_dataset(tf_datasetPath, split)

        all_pos = []
        all_node_type = []
        all_velocity = []
        all_cells = []
        filename = os.path.join(tf_datasetPath, split+'.dat')

        shape0, shape1 = 0, 0
        for index, d in enumerate(ds):
           velocity = d['velocity'].numpy()
           velocity = velocity.transpose(1, 0, 2) 
           N, T, D = velocity.shape
           shape0 += N
           shape1 = max(shape1, T)
           del velocity
        
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=(shape0, shape1, 2))

        write_shift  = 0
        for index, d in enumerate(ds):
            pos_ = d['mesh_pos'].numpy()
            node_type_ = d['node_type'].numpy()
            velocity = d['velocity'].numpy()
            cells_ = d['cells'].numpy()

            pos = pos_[0].copy() # same for all time steps,  step 0 only
            node_type = node_type_[0].copy() # same for all time steps,  step 0 only
            cells = cells_[0].copy() # same for all time steps,  step 0 only
            del pos_ # memory efficient operation
            del node_type_ # memory efficient operation
            del cells_ # memory efficient operation

            print(pos.shape, node_type.shape, velocity.shape, cells.shape)
          
            all_pos.append(pos)
            all_node_type.append(node_type)
            all_cells.append(cells)
            
            velocity = velocity.transpose(1, 0, 2)
            fp[write_shift:write_shift+velocity.shape[0]] = velocity

            fp.flush()
            write_shift += velocity.shape[0]
            del velocity
        del fp

        indices = [i.shape[0] for i in all_pos]
        indices = np.cumsum(indices)
        indices = np.insert(indices, 0, 0)

        cindices = [i.shape[0] for i in all_cells]
        cindices = np.cumsum(cindices)
        cindices = np.insert(cindices, 0, 0)

        all_pos = np.concatenate(all_pos, axis=0)
        all_node_type = np.concatenate(all_node_type, axis=0)
        all_cells = np.concatenate(all_cells, axis=0)

        np.savez_compressed(os.path.join(tf_datasetPath, split+'.npz'),
                            pos=all_pos,
                            node_type=all_node_type,
                            cells=all_cells,
                            indices=indices,
                            cindices=cindices,
                            all_velocity_shape=(shape0, shape1, 2)
        )