import os
class pi2pix():
    def __init__(self):
        pass

    def tb_generate(self):
            """
             Put NLM-MontgomeryCXRSet resized dataset into  pix2pix-pytorch/dataset/pix2pix/a
             Put  masked and resized imaged under pix2pix-pytorch/dataset/pix2pix/b
             Train.py will map the input from folder input folder a to otputfolder b
             test.py will generate mask output for given chest Xray images
                         
             output -- masks images for ChinaSet_AllFiles dataset
                                        """
            os.chdir("pix2pix-pytorch")
            #os.system('python train.py --dataset tb  --direction="a2b"')
            os.system('python test.py --dataset tb_test --direction="a2b" --cuda')
            return

train_pix2pix=pi2pix()
train_pix2pix.tb_generate()


















