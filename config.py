class DefaultConfigs(object):
    train_data = "./human_protein/train/" # where is your train data
    test_data = "./human_protein/test/"   # your test data
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    submit = "./submit/"
    model_name = "bninception_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.03
    batch_size = 32
    epochs = 50

config = DefaultConfigs()
