import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os
import cv2
import warnings




class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beyin Tümörü Teşhisi")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        
        self.training_label = QLabel(self)
        self.training_label.setAlignment(Qt.AlignCenter)
        
        

        self.result_layout = QGridLayout()

        self.detect_button = QPushButton("Teşhis Et", self)
        self.detect_button.clicked.connect(self.detect_tumor)
        
        
        self.layout.addWidget(self.image_label)
        self.layout.addLayout(self.result_layout)
        self.layout.addWidget(self.detect_button)
    
            
        
    def detect_tumor(self):
        
        warnings.filterwarnings("ignore")

        #DATA HAZIRLAMA VE TOPLAMA
        path = os.listdir("/Users/ouzha/Desktop/Brain_Tumor/Training/")
        classes = {"no_tumor":0, "pituitary_tumor":1}

        X = [] #görüntüleri tutacak olan boş dizi
        Y = [] #görüntülerin sınıflarını tutacak olan boş dizi

        #her bir sınıftaki görüntüler için görüntüyü okur
        #gri seviyeye çeker ve yeniden boyutlandırır.
        for cls in classes:
            pth = "/Users/ouzha/Desktop/Brain_Tumor/Training/"+cls
            for i in os.listdir(pth):
                img = cv2.imread(pth+'/'+i,0)
                img = cv2.resize(img,(200,200))

                X.append(img), Y.append(classes[cls]) #görüntüleri X'e ve görüntülerin sınıf değerlerini de Y'ye ekler.

        np.unique(Y) #Y değerlerini sıralı bir numpy dizisi haline getirir.

        X = np.array(X) #X değerlerini numpy array yapar
        Y = np.array(Y) #Y değerlerini numpy array yapar
        
        print("Sınıf Görüntü Sayısı =",pd.Series(Y).value_counts()) #Hangi sınıftan kaç adet görüntü olup olmadığını belirler

        print("Toplam Görüntü Sayısı =",X.shape) #Toplam kaç adet görüntü olduğunu yazar ve boyutunu yazar

        #DATA HAZIRLAMA
        #sklearn için data 3 boyuttan iki boyuta düşürülmeli
        #burada o işlemi yapıyoruz

        X_updated = X.reshape(len(X),-1)

        print("Yeniden Düzenleme Sonrası Görüntü Değerleri =",X_updated.shape)

        xtrain, xtest, ytrain, ytest = train_test_split(X_updated,Y,random_state=10,test_size=.20)
        print("xtrain degerleri{}, xtest degerler{}".format(xtrain.shape, xtest.shape))

        print("xtrain.max{}, xtrain.min{}".format(xtrain.max(), xtrain.min()))
        print("xtest.max{}, xtest.min{}".format(xtest.max(), xtest.min()))

        xtrain = xtrain/255
        xtest = xtest/255

        print("xtrain.max{}, xtrain.min{}".format(xtrain.max(), xtrain.min()))
        print("xtest.max{}, xtest.min{}".format(xtest.max(), xtest.min()))

        print("xtrain degerleri{}, xtest degerler{}".format(xtrain.shape, xtest.shape))

        pca = PCA(.98)
        pca_train = xtrain
        pca_test = xtest

        #TRAIN MODEL
        lg = LogisticRegression(C=0.1)
        lg.fit(pca_train, ytrain)

        sv = SVC()
        sv.fit(pca_train, ytrain)

        #EVALUATION 
        print("Training Score:", lg.score(pca_train, ytrain))
        print("Testing Score:",lg.score(pca_test, ytest))

        print("Training Score:", sv.score(pca_train, ytrain))
        print("Testing Score:", sv.score(pca_test, ytest))

        #PREDICTION
        pred = sv.predict(pca_test)
        np.where(ytest != pred)

        pred[36]
        ytest[36]

        #TEST MODEL
        dec = {0: "Tümör Yok", 1:"Tümör Var"}


        
        count=0
        
        self.clear_result_layout()
        if os.listdir("/Users/ouzha/Desktop/Brain_Tumor/Testing/idk_tumor/"):
            count = len(os.listdir("/Users/ouzha/Desktop/Brain_Tumor/Testing/idk_tumor/"))
       
        if(os.listdir("/Users/ouzha/Desktop/Brain_Tumor/Testing/idk_tumor/") != 0):
            for a in os.listdir("/Users/ouzha/Desktop/Brain_Tumor/Testing/idk_tumor/"):
             count = count + 1

        if(os.listdir("/Users/ouzha/Desktop/Brain_Tumor/Testing/idk_tumor/")):
            
            plt.figure(figsize = (12,8))
            p = os.listdir("/Users/ouzha/Desktop/Brain_Tumor/Testing/")
            c=1
            row = 0
            col = 0
            
            for j in os.listdir("/Users/ouzha/Desktop/Brain_Tumor/Testing/idk_tumor/")[:count]:
                
                plt.subplot(3,3,c)


                img_path = "/Users/ouzha/Desktop/Brain_Tumor/Testing/idk_tumor/" + j


                img = cv2.imread("/Users/ouzha/Desktop/Brain_Tumor/Testing/idk_tumor/"+j,0)
                img1 = cv2.resize(img, (200,200))
                img1 = img1.reshape(1,-1)/255
                p = sv.predict(img1)
                result_text = dec[p[0]]
                plt.title(dec[p[0]])
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.show()
                
                
                
                image_layout = QVBoxLayout()
                image_label = QLabel(self)
                pixmap = QPixmap(img_path).scaled(200, 200, Qt.KeepAspectRatio)
                image_label.setPixmap(pixmap)
                image_label.setAlignment(Qt.AlignCenter)
                image_layout.addWidget(image_label)

                result_label = QLabel(self)
                result_label.setText(result_text)
                result_label.setAlignment(Qt.AlignCenter)
                image_layout.addWidget(result_label)

                self.result_layout.addLayout(image_layout, row, col)
                col += 1
                if col == 3:
                    row += 1
                    col = 0
                
               
                
                c+=1
                

            self.result_label.setText("Teşhis tamamlandı...")
    

    def clear_result_layout(self):
        for i in reversed(range(self.result_layout.count())):
            layout_item = self.result_layout.itemAt(i)
            if layout_item is not None:
                layout_item.layout().deleteLater()
            self.result_layout.removeItem(layout_item)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    