import tensorflow as tf
import config
print(config)


def load_nsynth(path,cache1,cache2):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(path)
    length = 64000
    #dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
    #dataset = dataset.apply(
    #    tf.data.experimental.parallel_interleave(
    #        tf.data.TFRecordDataset, cycle_length=20, sloppy=True))
  
    def _parse_nsynth(record):
      
      out = 8192
      """Parsing function for NSynth dataset."""
      features = {
            'pitch': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'audio': tf.io.FixedLenFeature([length], dtype=tf.float32),
            'qualities': tf.io.FixedLenFeature([10], dtype=tf.int64),
            'instrument_str':tf.io.FixedLenFeature([1],dtype=tf.string),
            'instrument_source': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'instrument_family': tf.io.FixedLenFeature([1], dtype=tf.int64),
          }

      example = tf.io.parse_single_example(record, features)
      wave, pitch = example['audio'], example['pitch']
      #instrument_num = tf.strings.to_number(tf.strings.split(example["instrument_str"],"_")[-1],out_type="int32")
      instrument_str = tf.strings.to_number(tf.strings.split(example["instrument_str"],"_")[0][-1],out_type="int32")
      qualities = example['qualities']
      pitch = tf.cast(pitch,dtype="int32")
      #pitch = tf.one_hot(pitch,depth=128)

      #pitch = tf.keras.utils.to_categorical(pitch, num_classes=128)
      family = tf.cast(example['instrument_family'],dtype="int32")
      #family = tf.one_hot(family,depth=11)

      inum = tf.reshape(instrument_str,shape=(1,))
      return wave,pitch,family, inum,qualities
    
    def drop_quality(a,p,f,i,q):
        return a,p,f,i


    def mu_law(x, mu=255, int8=False):
        """A TF implementation of Mu-Law encoding.
            Args:
            x: The audio samples to encode.
            mu: The Mu to use in our Mu-Law.
            int8: Use int8 encoding.
            Returns:
            out: The Mu-Law encoded int8 data.
        """

        x = x / (tf.reduce_max(x,axis=-1,keepdims=True))
        x = x / (1+1e-5)
        out = tf.sign(x) * tf.math.log(1 + mu * tf.abs(x)) / tf.math.log(1. + mu)
        out = tf.floor(out * 128)
        if int8:
            out = tf.cast(out, tf.int8)
        return out + 128.
    def inv_mu_law(x, mu=255):
      """A TF implementation of inverse Mu-Law.

      Args:
        x: The Mu-Law samples to decode.
        mu: The Mu we used to encode these samples.

      Returns:
        out: The decoded data.
      """
      x = tf.cast(x, tf.float32) - 128.
      out = (x + 0.5) * 2. / (mu + 1)
      out = tf.sign(out) / mu * ((1 + mu)**tf.abs(out) - 1)
      out = tf.where(tf.equal(x, 0), x, out)
      return out

    def mix(b,s):
        return (b[0]+s[0])/2.,b,s

    def to_categorical(a,p,f,i):
        
        
        pitch = tf.one_hot(p[0],depth=128)
        family = tf.one_hot(f[0],depth=11)
        return a,pitch,family,i
    
    def random_crop(a,p,f,i):
        start = tf.random.uniform(shape=(),minval=0,maxval= 4096,dtype=tf.int32)
        audio = tf.slice(a,[start,],[8192,])
        return audio,p,f,i
    
    def preprocess(w,l,f,i):
        w = mu_law(w)
        w = tf.one_hot(tf.cast(w,"int32"),256)
        return w,l,f,i

    def normlize(x):
      x = x / (tf.reduce_max(x,axis=-1,keepdims=True))
      x = x / (1+1e-5)
      
        
    def preprocess(m,b,s):
        #m_enc = tf.one_hot(tf.cast(mu_law(m),"int32"),256)
        #b_enc = tf.one_hot(tf.cast(mu_law(b[0]),"int32"),256)
        #s_enc = tf.one_hot(tf.cast(mu_law(s[0]),"int32"),256)
        
        #m_enc = tf.expand_dims(tf.cast(mu_law(m),"float32"),axis=1)
        #b_enc = tf.expand_dims(tf.cast(mu_law(b[0]),"float32"),axis=1)
        #s_enc = tf.expand_dims(tf.cast(mu_law(s[0]),"float32"),axis=1)

        m_enc = tf.expand_dims(m,axis=1)
        b_enc = tf.expand_dims(b[0],axis=1)
        s_enc = tf.expand_dims(s[0],axis=1)

        return m_enc,(b_enc,)+b[1:],(s_enc,)+s[1:]


    
    
    dataset = dataset.map(_parse_nsynth)

    #dataset = dataset.filter(lambda w,l,i:tf.reduce_all(tf.equal(1,i)))
    dataset = dataset.filter(lambda a,l,f,i,q:tf.equal(q[7],0)).map(drop_quality)
    dataset1 = dataset.filter(lambda a,l,f,i,: tf.equal(f,0)[0]).cache(cache1).map(to_categorical)
    dataset2 = dataset.filter(lambda a,l,f,i,: tf.equal(f,6)[0]).cache(cache2).map(to_categorical)
    dataset1 = dataset1.shuffle(1000,reshuffle_each_iteration=True).repeat()
    dataset2 = dataset2.shuffle(1000,reshuffle_each_iteration=True).repeat()
    dataset1 = dataset1.map(random_crop).prefetch(AUTOTUNE)
    dataset2 = dataset2.map(random_crop).prefetch(AUTOTUNE)
    
    bs_dataset= tf.data.Dataset.zip((dataset1,dataset2))
    mixed_dataset = bs_dataset.map(mix)
    mixed_dataset = mixed_dataset.map(preprocess)

   
    #dataset = dataset.map(preprocess)
    
    
    
    return mixed_dataset