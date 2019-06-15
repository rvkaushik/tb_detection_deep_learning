import cv2,os,imutils

class mask():
    def __init__(self):
        pass
    def merge_masks(self,path):
        left_masks=self.get_files(path + r"/leftMask")
        #print (left_masks)
        right_masks=self.get_files(path + r"/rightMask")
        output_path=path + r"/masks_merged"
        os.mkdir(output_path)
        self.merge(left_masks,right_masks)


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
            merged=imutils.resize(left+right,width=400)
            out=left_mask.replace("leftMask","masks_merged")
            cv2.imwrite(out,merged)
        return

path="/home/ravi/code/ravi/github_projects/tb_detection_deep_learning/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask"

tb_mask=mask()
tb_mask.merge_masks(path)




