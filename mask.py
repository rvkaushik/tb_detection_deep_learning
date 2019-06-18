import cv2,os,imutils,shutil

class mask():
    def __init__(self,path_NLM,path_CHINA,path_pix2pix):
        self.path_NLM=path_NLM
        self.path_CHINA=path_CHINA
        self.path_pix2pix=path_pix2pix
    def merge_masks(self,path):
        left_masks=self.get_files(path + r"/leftMask")
        #print (left_masks)
        right_masks=self.get_files(path + r"/rightMask")
        output_path=path + r"/masks_merged"
        #os.mkdir(output_path)
        #self.merge(left_masks,right_masks)
        nlm_files=self.get_files(self.path_NLM)
        china_files=self.get_files(self.path_CHINA)
        nlm_masks=self.get_files(output_path)
        self.create_test_train(nlm_files,self.path_pix2pix,"a")
        self.create_test_train(nlm_masks,self.path_pix2pix,"b")
        self.create_test_train(china_files,self.path_pix2pix,"a")
       # self.resize_all(nlm_files)
        #self.resize_all(china_files)


        return

    def get_files(self,path):
        files=[]
        for (dirpath, dirnames, filenames) in os.walk(path):
            for filename in filenames:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    files.append(os.path.join(path,filename))
        #print(files)
        return files

    def merge(self,left_masks , right_masks):
        for (left_mask,right_mask) in zip(left_masks,right_masks):
            left=cv2.imread(left_mask,1)
            right=cv2.imread(right_mask,1)
            merged=imutils.resize(left+right,width=227,height=227)
            out=left_mask.replace("leftMask","masks_merged")
            cv2.imwrite(out,merged)
        return

    def resize_all(self,images):
        for image in images:
            img=cv2.imread(image,1)
            resized=cv2.resize(img,(227,227))
            cv2.imwrite(image,resized)

        return

    def create_test_train(self,images,path_pix2pix,typ,split_data=True):
        if split_data:
            split=int(len(images)*(0.8))
            test=images[split:]
            dataset_path_test=path_pix2pix+r"/dataset/tb/test/{}".format(typ)
            if not os.path.exists(dataset_path_test):
                os.makedirs(dataset_path_test)
            self.copy_files(test,dataset_path_test)
        else :
            split=int(len(images))
        
        train=images[:split]
        dataset_path_train=path_pix2pix+r"/dataset/tb/train/{}".format(typ)
        if not os.path.exists(dataset_path_train):
            os.makedirs(dataset_path_train)
        self.copy_files(train,dataset_path_train)

        return
    def copy_files(self,src,dst):
        for img in src:
            shutil.copy(img,dst)


path="/home/ravi/code/ravi/github_projects/tb_detection_deep_learning/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask"


path_NLM="/home/ravi/code/ravi/github_projects/tb_detection_deep_learning/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png"


path_CHINA="/home/ravi/code/ravi/github_projects/tb_detection_deep_learning/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask"

path_pix2pix="/home/ravi/code/ravi/github_projects/tb_detection_deep_learning/pix2pix-pytorch"

tb_mask=mask(path_NLM,path_NLM,path_pix2pix)
tb_mask.merge_masks(path)






