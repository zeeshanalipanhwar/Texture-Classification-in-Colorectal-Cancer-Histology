class Configs:
    def __init__(self):
        # Name of the dataset
        self.DATASETNAME = "Kather_texture_2016_image_tiles_5000"
        
        # Google drive link to the dataset
        self.DATALINK = "/content/drive/My Drive/" + self.DATASETNAME

        # Batch size for training and testing of a model
        self.BATCHSIZE = 32
        
        # Number of epochs for training of a model
        self.EPOCHS = 1
        
        # Number of classes in the dataset
        self.NUMCLASSES = 8
        
        # Class names in the dataset
        self.CLASSES = ['ADIPOSE', 'COMPLEX', 'DEBRIS', 'EMPTY', 'LYMPHO', 'MUCOSA', 'STROMA', 'TUMOR']

        # Model training setting
        self.LOSS = "sparse_categorical_crossentropy"
        self.OPTIMIZER = "adam"
        self.METRICS = ["accuracy"]

    def display(self):
        print ("Name of the dataset: {}.".format(self.DATASETNAME)
        
        print ("Google drive link to the dataset: {}.".format(self.DATALINK)

        print ("Batch size for training and testing of a model: {}.".format(self.BATCHSIZE)
        
        print ("Number of epochs for training of a model: {}.".format(self.EPOCHS)
        
        print ("Number of classes in the dataset: {}.".format(self.NUMCLASSES)
        
        print ("Class names in the dataset: {}.".format(self.CLASSES)

        print ("Model training setting")
        print ("loss: {}.".format(self.LOSS)
        print ("optimizer: {}.".format(self.OPTIMIZER)
        print ("metrics: {}.".format(self.METRICS)
