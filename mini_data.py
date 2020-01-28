
import tensorflow as tf

def load_minidata(path):
    feature_description = {
     "audio":tf.io.FixedLenFeature([64000], tf.float32), 
     "pitch":tf.io.FixedLenFeature([128], tf.float32),
     "instrument":tf.io.FixedLenFeature([11], tf.float32)
    }
    def _parse(example_proto):
        data =  tf.io.parse_single_example(example_proto, feature_description)
        return data["audio"],data["pitch"],data["instrument"]
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(_parse)
    return dataset

def make_dataset(path1,path2):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    def random_crop(a,p,i):
        start = tf.random.uniform(shape=(),minval=0,maxval= 4096,dtype=tf.int32)
        audio = tf.slice(a,[start,],[8192,])
        return audio,p,i
    
    def mix(inst1,inst2):
        return (inst1[0]+inst2[0])/2.,inst1,inst2
    def preprocess(m,b,s):
        m_enc = tf.expand_dims(m,axis=1)
        b_enc = tf.expand_dims(b[0],axis=1)
        s_enc = tf.expand_dims(s[0],axis=1)

        return m_enc,(b_enc,)+b[1:],(s_enc,)+s[1:]
    dataset1 = load_minidata(path1)
    dataset2 = load_minidata(path2)
    dataset1 = dataset1.shuffle(1000,reshuffle_each_iteration=True).repeat()
    dataset2 = dataset2.shuffle(1000,reshuffle_each_iteration=True).repeat()
    dataset1 = dataset1.map(random_crop).prefetch(AUTOTUNE)
    dataset2 = dataset2.map(random_crop).prefetch(AUTOTUNE)
    
    mixed_dataset= tf.data.Dataset.zip((dataset1,dataset2))
    mixed_dataset = mixed_dataset.map(mix)
    mixed_dataset = mixed_dataset.map(preprocess)
    return mixed_dataset
    
if __name__ == "__main__":
    dataset = make_dataset("bass.tfrecord","flute.tfrecord")
    print(dataset)