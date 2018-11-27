from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.preprocessing import MultiLabelBinarizer
from imgaug import augmenters as iaa

# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# create dataset class
class HumanDataset(Dataset):
    def __init__(self,images_df,base_path,augument=True,mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        self.images_df = images_df.copy()
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x:base_path / x)
        self.mlb = MultiLabelBinarizer(classes = np.arange(0,config.num_classes))
        self.mlb.fit(np.arange(0,config.num_classes))
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        X = self.read_images(index)
        if not self.mode == "test":
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
            y  = np.eye(config.num_classes,dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        #X = T.Compose([T.ToPILImage(),T.ToTensor(),T.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])])(X)
        X = T.Compose([T.ToPILImage(),T.ToTensor()])(X)
        return X.float(),y


    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        #use only rgb channels
        if config.channels == 4:
            images = np.zeros(shape=(512,512,4))
        else:
            images = np.zeros(shape=(512,512,3))
        r = np.array(Image.open(filename+"_red.png")) 
        g = np.array(Image.open(filename+"_green.png")) 
        b = np.array(Image.open(filename+"_blue.png")) 
        y = np.array(Image.open(filename+"_yellow.png")) 
        images[:,:,0] = r.astype(np.uint8) 
        images[:,:,1] = g.astype(np.uint8)
        images[:,:,2] = b.astype(np.uint8)
        if config.channels == 4:
            images[:,:,3] = y.astype(np.uint8)
        images = images.astype(np.uint8)
        #images = np.stack(images,-1) 
        if config.img_height == 512:
            return images
        else:
            return cv2.resize(images,(config.img_weight,config.img_height))

    def augumentor(self,image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
