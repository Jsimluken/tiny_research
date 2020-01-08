import tensorflow as tf
from .model import *
from .dataset import *
import tqdm


def train_wavenet_pitch(strategy,LOGDIR,model_path,cache1,cache2):
  now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9)))
  str_time = now.strftime("%Y_%m_%d_%H_%M_%S")
  LOGDIR= f"{LOGDIR}_{str_time}"
  print(f"LOGDIR!! {LOGDIR}")
  with strategy.scope(): 
    summary_writer = tf.summary.create_file_writer(LOGDIR)
    loss_key = ["ge_pitch_loss","gd_audio_loss"]
    encoder = create_encoder()
    decoder = create_decoder()
    classifier = create_classifier()
    
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), encoder=encoder, decoder=decoder,classifier=classifier)
    manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)
    
    classifier_loss = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
    mse_loss = tf.losses.MeanSquaredError(reduction=tf.losses.Reduction.NONE)
    bin_loss = tf.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE)

    # Generator optimizers!!
    gec_optim = tf.optimizers.Adam(learning_rate=0.0006)
    gec_optim2 = tf.optimizers.Adam(learning_rate=0.0006)

    gd_audio_optim1 = tf.optimizers.Adam(learning_rate=0.0004)
    gd_audio_optim2 = tf.optimizers.Adam(learning_rate=0.0002)
    gd_audio_optim3 = tf.optimizers.Adam(learning_rate=0.0006)


    pitch_acc = tf.metrics.CategoricalAccuracy()
    #dec_acc = tf.metrics.CategoricalAccuracy()
    dec_acc = tf.metrics.MeanSquaredError()

    class Reduction(Enum):
        NONE = 0
        SUM = 1
        MEAN = 2
        CONCAT = 3

    def distrtibuted(*reduction_flags):
        def _decorator(fun):
            def per_replica_reduction(z, flag):
                if flag == Reduction.NONE:
                    return z
                elif flag == Reduction.SUM:
                    return strategy.reduce(tf.distribute.ReduceOp.SUM, z, axis=None)
                elif flag == Reduction.MEAN:
                    return strategy.reduce(tf.distribute.ReduceOp.MEAN, z, axis=None)
                elif flag == Reduction.CONCAT:
                    z_list = strategy.experimental_local_results(z)
                    return tf.concat(z_list, axis=0)
                else:
                    raise NotImplementedError()

            @tf.function
            def _decorated_fun(*args, **kwargs):
                fun_result = strategy.experimental_run_v2(fun, args=args, kwargs=kwargs)
    
                assert type(fun_result) is tuple
                return tuple((per_replica_reduction(fr, Reduction.SUM) for fr in fun_result[:-1])) + (per_replica_reduction(fun_result[-1],Reduction.CONCAT),)

                #return tuple((per_replica_reduction(fr, rf) for fr in fun_result))
            return _decorated_fun
        return _decorator

    @distrtibuted()
    def train_on_batch(X_mix,family,pitch,X_clean):
      dummy_instrument = tf.zeros_like(family)
      gt_real = tf.zeros((X_mix.shape[0],1))
      gt_fake = tf.ones((X_mix.shape[0],1))

      print("enc!!")
      with tf.GradientTape() as gec_tape,tf.GradientTape() as gd_tape,tf.GradientTape() as tape3, \
      tf.GradientTape() as tape4,tf.GradientTape() as tape5:
        encs = encoder(X_mix)
      print("Encoder cls")
      with gec_tape,gd_tape:
        
        ge_pitch,_,_ = classifier([encs[-1],family])
        ge_pitch_loss = classifier_loss(pitch,ge_pitch)
        ge_pitch_loss = tf.reduce_sum(ge_pitch_loss,keepdims=True) * (1. / batch_size)
      print("Encoder cls end!!")
      ge_pitch_param = [encoder.trainable_weights,classifier.trainable_weights]
      ge_pitch_gradient = gec_tape.gradient(ge_pitch_loss,ge_pitch_param)
      gec_optim.apply_gradients(zip(ge_pitch_gradient[0],ge_pitch_param[0]))
      gec_optim2.apply_gradients(zip(ge_pitch_gradient[1],ge_pitch_param[1]))

      pitch_acc.update_state(pitch,ge_pitch)
      
      print("Decoder!!")
      with gd_tape,tape3,tape4,tape5:
        gd_audio = decoder([encs[::-1],ge_pitch,family])
        gd_audio_loss = mse_loss(X_clean,gd_audio)
        gd_audio_loss = tf.reduce_sum(tf.reduce_sum(gd_audio_loss,axis=-1),keepdims=True) * (1. / batch_size)
      gd_audio_param = [encoder.trainable_weights,classifier.trainable_weights,decoder.trainable_weights]
      gd_audio_gradient = gd_tape.gradient(gd_audio_loss,gd_audio_param)
      gd_audio_optim1.apply_gradients(zip(gd_audio_gradient[0],gd_audio_param[0]))
      gd_audio_optim2.apply_gradients(zip(gd_audio_gradient[1],gd_audio_param[1]))
      gd_audio_optim3.apply_gradients(zip(gd_audio_gradient[2],gd_audio_param[2]))
      dec_acc.update_state(X_clean,gd_audio)

     
      #df_optim1.apply_gradients(zip())
      
      return (ge_pitch_loss,gd_audio_loss,gd_audio)
      
    def create_dummy_data():
      mix_audio = tf.zeros((1,8192,256),dtype="float32")
      audio = tf.zeros((1,8192,256),dtype="float32" )
      pitch = tf.zeros((1,128),dtype="float32")
      family = tf.zeros((1,11),dtype="float32")

      return mix_audio,pitch,family,audio
    batch_size = 512
    dataset = load_nsynth()
    #dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = strategy.experimental_distribute_dataset(dataset)
    print("starting!!")
    
    def to_audio(x):
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
        #mu_law = tf.argmax(x,axis=-1)
        mu_law = tf.round(x[:,:,0])
        
        return tf.expand_dims(inv_mu_law(mu_law),axis=-1)

    with tqdm.tqdm(dataset) as pbar:
      #manager.
      pitch_acc.reset_states()
      dec_acc.reset_states()
      for step,(m,b,s) in enumerate(pbar):
        X_mix = m
        X_clean = b[0]
        pitch = b[1]
        family = b[2]
        res = train_on_batch(X_mix,family,pitch,X_clean)
        pbar_post_dic = {}
        for i,k in enumerate(loss_key):
            pbar_post_dic[k] = res[i].numpy()[0]
        

        pbar.set_description(f"{step} step, pitch {pitch_acc.result():.2f} family {dec_acc.result():.2f}")
        loss = tf.reduce_sum(res[:-1])
        pbar_post_dic["loss"] = loss.numpy()
        pbar.set_postfix(pbar_post_dic)
        manager.save()
        
        with summary_writer.as_default():
            for key in pbar_post_dic.keys():
                val = pbar_post_dic[key]
                tf.summary.scalar(key,val,step=step,)
            tf.summary.scalar("pitch_acc",pitch_acc.result(),step=step)
            tf.summary.scalar("dec_acc",dec_acc.result(),step=step)
            if step %100 == 0:
                mix_audio = strategy.experimental_local_results(X_mix)[0][0:10]
                clean_audio = strategy.experimental_local_results(X_clean)[0][0:10]
                out_audio = res[-1][0:10]
                mix_audio = to_audio(mix_audio)
                clean_audio = to_audio(clean_audio)
                out_audio = to_audio(out_audio)
                tf.summary.audio("mix_audio",mix_audio,16000,step=step,max_outputs=10)
                tf.summary.audio("clean_audio",clean_audio,16000,step=step,max_outputs=10)
                tf.summary.audio("out_audio",out_audio,16000,step=step,max_outputs=10)
                
    print("end")
