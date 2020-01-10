from model import *
from dataset import *
import tqdm

def train(strategy,LOGDIR,model_path):
  now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9)))
  str_time = now.strftime("%Y_%m_%d_%H_%M_%S")
  LOGDIR= f"{LOGDIR}_{str_time}"
  print(f"LOGDIR!! {LOGDIR}")
  with strategy.scope(): 
    summary_writer = tf.summary.create_file_writer(LOGDIR)
    loss_key = ["ge_pitch_loss","gd_audio_loss","gdc_pitch_loss","gdc_family_loss","gdc_fake_loss",
              "drp_pitch_loss","drp_family_loss","drp_fake_loss","dfp_pitch_loss","dfp_family_loss","dfp_fake_loss"]
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
    gec_optim_e = tf.optimizers.Adam(learning_rate=0.00004)
    gec_optim_c = tf.optimizers.Adam(learning_rate=0.00004)

    gd_audio_optim_e = tf.optimizers.Adam(learning_rate=0.00005)
    gd_audio_optim_c = tf.optimizers.Adam(learning_rate=0.00001)
    gd_audio_optim_d = tf.optimizers.Adam(learning_rate=0.00005)

    gdc_optim1_e = tf.optimizers.Adam(learning_rate=0.00003)
    gdc_optim1_c = tf.optimizers.Adam(learning_rate=0.00003)
    gdc_optim1_d = tf.optimizers.Adam(learning_rate=0.00003)
    
    gdc_optim2_e = tf.optimizers.Adam()
    gdc_optim2_c = tf.optimizers.Adam()
    gdc_optim2_d = tf.optimizers.Adam()
    
    gdc_optim3_e = tf.optimizers.Adam(learning_rate=0.00004)
    gdc_optim3_c = tf.optimizers.Adam(learning_rate=0.00004)
    gdc_optim3_d = tf.optimizers.Adam(learning_rate=0.00004)

    #Discrimonator optimizers!!
    dr_optim1_e = tf.optimizers.Adam(learning_rate=0.00005)
    dr_optim1_c = tf.optimizers.Adam(learning_rate=0.00005)
    dr_optim1_d = tf.optimizers.Adam(learning_rate=0.00005)
    
    dr_optim2_e = tf.optimizers.Adam()
    dr_optim2_c = tf.optimizers.Adam()
    dr_optim2_d = tf.optimizers.Adam()
    
    dr_optim3_e = tf.optimizers.Adam(learning_rate=0.00001)
    dr_optim3_c = tf.optimizers.Adam(learning_rate=0.0001)

    df_optim1 = tf.optimizers.Adam(learning_rate=0.00004)
    df_optim2 = tf.optimizers.Adam(learning_rate=0.00004)
    df_optim3 = tf.optimizers.Adam(learning_rate=0.00004)

    pitch_acc = tf.metrics.CategoricalAccuracy()
    dec_acc = tf.metrics.CategoricalAccuracy()
    dr_acc = tf.metrics.BinaryAccuracy()
    df_acc = tf.metrics.BinaryAccuracy()
    gf_acc = tf.metrics.BinaryAccuracy()
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
      gec_optim_e.apply_gradients(zip(ge_pitch_gradient[0],ge_pitch_param[0]))
      gec_optim_c.apply_gradients(zip(ge_pitch_gradient[1],ge_pitch_param[1]))

      pitch_acc.update_state(pitch,ge_pitch)
      
      print("Decoder!!")
      with gd_tape,tape3,tape4,tape5:
        gd_audio = decoder([encs[::-1],ge_pitch,family])
        gd_audio_loss = classifier_loss(X_clean,gd_audio)
        gd_audio_loss = tf.reduce_sum(tf.reduce_sum(gd_audio_loss,axis=-1),keepdims=True) * (1. / batch_size)
      gd_audio_param = [encoder.trainable_weights,classifier.trainable_weights,decoder.trainable_weights]
      gd_audio_gradient = gd_tape.gradient(gd_audio_loss,gd_audio_param)
      gd_audio_optim_e.apply_gradients(zip(gd_audio_gradient[0],gd_audio_param[0]))
      gd_audio_optim_c.apply_gradients(zip(gd_audio_gradient[1],gd_audio_param[1]))
      gd_audio_optim_d.apply_gradients(zip(gd_audio_gradient[2],gd_audio_param[2]))
      dec_acc.update_state(X_clean,gd_audio)

      print("GDC!")
      gdc_params = [encoder.trainable_weights,classifier.trainable_weights,decoder.trainable_weights]
      with tape3,tape4,tape5:
        gd_enced = encoder(gd_audio)
        gdc_pitch,gdc_family,gdc_fake = classifier([gd_enced[-1],family])
      with tape3:
        gdc_pitch_loss = classifier_loss(pitch,gdc_pitch)
        gdc_pitch_loss = tf.reduce_sum(gdc_pitch_loss,keepdims=True) * (1. / batch_size)
      #gdc_pitch_param = []
      gdc_pitch_gradient = tape3.gradient(gdc_pitch_loss,gdc_params)
      with tape4:
        gdc_family_loss = classifier_loss(family,gdc_family)
        gdc_family_loss = tf.reduce_sum(gdc_family_loss,keepdims=True) * (1. / batch_size)
      gdc_family_gradient = tape4.gradient(gdc_family_loss,gdc_params)
      with tape5:
        gdc_fake_loss = bin_loss(gt_real,gdc_fake)
        gdc_fake_loss = tf.reduce_sum(gdc_fake_loss,keepdims=True) * (1. / batch_size)
      gdc_fake_gradient = tape5.gradient(gdc_fake_loss,gdc_params)
      gf_acc.update_state(gt_real,gdc_fake)

      gdc_optim1_e.apply_gradients(zip(gdc_pitch_gradient[0],gdc_params[0]))
      gdc_optim1_c.apply_gradients(zip(gdc_pitch_gradient[1],gdc_params[1]))
      gdc_optim1_d.apply_gradients(zip(gdc_pitch_gradient[2],gdc_params[2]))

      #gdc_optim2_e.apply_gradients(zip(gdc_family_loss[0],gdc_params[0]))
      #gdc_optim2_c.apply_gradients(zip(gdc_family_loss[1],gdc_params[1]))
      #gdc_optim2_d.apply_gradients(zip(gdc_family_loss[2],gdc_params[2]))

      gdc_optim3_e.apply_gradients(zip(gdc_fake_gradient[0],gdc_params[0]))
      gdc_optim3_c.apply_gradients(zip(gdc_fake_gradient[1],gdc_params[1]))
      gdc_optim3_d.apply_gradients(zip(gdc_fake_gradient[2],gdc_params[2]))
      

      # discriminator!!
      print("d_enc")
      dr_params = [encoder.trainable_weights,classifier.trainable_weights]
      with tf.GradientTape() as dr_tape1,tf.GradientTape() as dr_tape2,tf.GradientTape() as dr_tape3:
        enced_xc = encoder(X_clean)
        drp_pitch,drp_family,drp_fake = classifier([enced_xc[-1],family])
      #dr_params = [encoder.trainable_weights,classifier.trainable_weights]
      with dr_tape1:
        drp_pitch_loss = classifier_loss(pitch,drp_pitch)
        drp_pitch_loss = tf.reduce_sum(drp_pitch_loss,keepdims=True) * (1. / batch_size)
      drp_pitch_gradient = dr_tape1.gradient(drp_pitch_loss,dr_params)

      with dr_tape2:
        drp_family_loss = classifier_loss(family,drp_family)
        drp_family_loss = tf.reduce_sum(drp_family_loss,keepdims=True) * (1. / batch_size)
      drp_family_gradient = dr_tape2.gradient(drp_family_loss,dr_params)

      with dr_tape3:
        drp_fake_loss = bin_loss(gt_real,drp_fake)
        drp_fake_loss = tf.reduce_sum(drp_fake_loss,keepdims=True) * (1. / batch_size)
      drp_fake_gradient = dr_tape3.gradient(drp_fake_loss,dr_params)
      dr_acc.update_state(gt_real,drp_fake)

      dr_optim1_e.apply_gradients(zip(drp_pitch_gradient[0],dr_params[0]))
      dr_optim1_c.apply_gradients(zip(drp_pitch_gradient[1],dr_params[1]))

      #dr_optim2_e.apply_gradients(zip(drp_family_gradient[0],dr_params[0]))
      #dr_optim2_c.apply_gradients(zip(drp_family_gradient[1],dr_params[1]))

      dr_optim3_e.apply_gradients(zip(drp_fake_gradient[0],dr_params[0]))
      dr_optim3_c.apply_gradients(zip(drp_fake_gradient[1],dr_params[1]))



      #discriminator fake!!
      print("dscriminator_fake!!")
      df_params = [classifier.trainable_weights]
      for i in range(2):
          with tf.GradientTape() as df_tape1,tf.GradientTape() as df_tape2,tf.GradientTape() as df_tape3:
            dfp_pitch,dfp_family,dfp_fake = classifier([gd_enced[-1],family])
          print("df_pitch!!")
          with df_tape1:
            dfp_pitch_loss = classifier_loss(pitch,dfp_pitch)
            dfp_pitch_loss = tf.reduce_sum(dfp_pitch_loss,keepdims=True) * (1. / batch_size)
          df_pitch_gradient = df_tape1.gradient(dfp_pitch_loss,df_params)
          print("df_family!!")
          with df_tape2:
            dfp_family_loss =classifier_loss(family,dfp_family)
            dfp_family_loss = tf.reduce_sum(dfp_family_loss,keepdims=True) * (1. / batch_size)
          df_family_gradient = df_tape2.gradient(dfp_family_loss,df_params)
          with df_tape3:
            dfp_fake_loss = bin_loss(gt_fake,dfp_fake)
            dfp_fake_loss = tf.reduce_sum(dfp_fake_loss,keepdims=True) * (1. / batch_size)
          df_fake_gradient = df_tape3.gradient(dfp_fake_loss,df_params)
          df_acc.update_state(gt_fake,dfp_fake)

          df_optim1.apply_gradients(zip(df_pitch_gradient[0],df_params[0]))
          df_optim2.apply_gradients(zip(df_family_gradient[0],df_params[0]))
          df_optim3.apply_gradients(zip(df_fake_gradient[0],df_params[0]))
          print("df end!!")
        
      

      #df_optim1.apply_gradients(zip())
      
      return (ge_pitch_loss,gd_audio_loss,gdc_pitch_loss,gdc_family_loss,gdc_fake_loss,
              drp_pitch_loss,drp_family_loss,drp_fake_loss,dfp_pitch_loss,dfp_family_loss,dfp_fake_loss,
              gd_audio)
      
    def create_dummy_data():
      mix_audio = tf.zeros((1,8192,256),dtype="float32")
      audio = tf.zeros((1,8192,256),dtype="float32" )
      pitch = tf.zeros((1,128),dtype="float32")
      family = tf.zeros((1,11),dtype="float32")

      return mix_audio,pitch,family,audio

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
        mu_law = tf.argmax(x,axis=-1)
        return tf.expand_dims(inv_mu_law(mu_law),axis=-1)

    batch_size = 512
    dataset = load_nsynth()
    #dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = strategy.experimental_distribute_dataset(dataset)
    print("starting!!")

    with tqdm.tqdm(dataset) as pbar:
      pitch_acc.reset_states()
      dec_acc.reset_states()
      gf_acc.reset_states()
      dr_acc.reset_states()
      df_acc.reset_states()  
      for step,(m,b,s) in enumerate(pbar):
        X_mix = m
        X_clean = b[0]
        pitch = b[1]
        family = b[2]
        res = train_on_batch(X_mix,family,pitch,X_clean)
        manager.save()
        
        pbar_post_dic = {}
        for i,k in enumerate(loss_key):
            pbar_post_dic[k] = res[i].numpy()[0]

        pbar.set_description(f"{step} step, pitch {pitch_acc.result():.2f} family {dec_acc.result():.2f}")
        loss = tf.reduce_sum(res[:-1])
        pbar_post_dic["loss"] = loss.numpy()
        pbar_post_dic["gf_acc"] = gf_acc.result().numpy()
        pbar_post_dic["dr_acc"] = dr_acc.result().numpy()
        pbar_post_dic["df_acc"] = df_acc.result().numpy()
        pbar.set_postfix(pbar_post_dic)
        with summary_writer.as_default():
            for key in pbar_post_dic.keys():
                val = pbar_post_dic[key]
                tf.summary.scalar(key,val,step=step)
            tf.summary.scalar("pitch_acc",pitch_acc.result(),step=step)
            tf.summary.scalar("dec_acc",dec_acc.result(),step=step)
            tf.summary.scalar("gf_acc",gf_acc.result(),step=step)
            tf.summary.scalar("dr_acc",dr_acc.result(),step=step)
            tf.summary.scalar("df_acc",df_acc.result(),step=step)
            if step %100 == 0:
                mix_audio = strategy.experimental_local_results(X_mix)[0][0:10]
                clean_audio = strategy.experimental_local_results(X_clean)[0][0:10]
                out_audio = res[-1][0:10]
                #mix_audio = to_audio(mix_audio)
                #clean_audio = to_audio(clean_audio)
                #out_audio = to_audio(out_audio)
                tf.summary.audio("mix_audio",mix_audio,16000,step=step,max_outputs=10)
                tf.summary.audio("clean_audio",clean_audio,16000,step=step,max_outputs=10)
                tf.summary.audio("out_audio",out_audio,16000,step=step,max_outputs=10)
         
        
    print("end")

