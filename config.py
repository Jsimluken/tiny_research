config = {
    "hyper":0
}

train_config = {
    "batch_size":512
}

model_config = {
    "num_layer":7,
    "initial_filter":32,
    "enc_filter":15,
}

dataset_config = {
    "use_mu_law":False,
    "inst1":"bass",
    "inst2":"organ",
    "sr":16000,
}

instruments = {
    "bass":0,
    "brass":1,
    "flute":2,
    "guitar":3,
    "keyboard":4,
    "mallet":5,
    "organ":6,
    "reed":7,
    "string":8,
    "synth_lead":9,
    "vocal":10
}

num2instrument = {
    0:	"bass",
    1:	"brass",
    2:	"flute",
    3:	"guitar",
    4:	"keyboard",
    5:	"mallet",
    6:	"organ",
    7:	"reed",
    8:	"string",
    9:	"synth_lead",
    10:	"vocal"
}

