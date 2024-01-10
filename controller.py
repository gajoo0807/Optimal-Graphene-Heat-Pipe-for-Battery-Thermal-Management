from msilib.schema import ComboBox
from operator import index, truediv
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QLabel, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from UI import Ui_MainWindow
from functools import partial
from surrogate import Surrogate_model
from regression_func import regressType
import cv2
import numpy as np
from numpy import genfromtxt
import sys
import os
import zipfile
import shutil

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #self.resize(1500, 1000)
        #self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.button_connect()
        self.df_name = ""
        self.ui.stackedWidget.setCurrentIndex(0)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        self.isShortedFlag = False
        
    def button_connect(self):
        self.ui.btn_close.clicked.connect(self.home_clicked)
        #page0
        self.ui.pushButton.clicked.connect(self.newModel_clicked)
        self.ui.pushButton_2.clicked.connect(self.importModel_clicked)
        #page1
        self.ui.pushButton_4.clicked.connect(self.open_clicked)
        self.ui.pushButton_9.clicked.connect(self.next_clicked_1)
        #page2
        self.ui.pushButton_5.clicked.connect(self.regression_clicked)
        self.ui.pushButton_8.clicked.connect(self.parameterOptimization_clicked)
        #page3
        self.ui.pushButton_3.clicked.connect(self.RBF_clicked)
        self.ui.pushButton_10.clicked.connect(self.KRG_clicked)
        self.ui.pushButton_11.clicked.connect(self.KPLS_clicked)
        #page4
        self.ui.pushButton_23.clicked.connect(self.next_clicked_4)
        #page5
        self.ui.pushButton_22.clicked.connect(self.next_clicked_5)
        #page6
        self.ui.pushButton_6.clicked.connect(self.continue_clicked)
        self.ui.pushButton_7.clicked.connect(self.saveModel_clicked)
        #page7
        self.ui.pushButton_12.clicked.connect(self.Lasso_clicked)
        self.ui.pushButton_24.clicked.connect(self.Ridge_clicked)
        #page8
        self.ui.pushButton_13.clicked.connect(self.next_clicked_8)
        #page10
        self.ui.pushButton_14.clicked.connect(self.open_clicked_10)
        self.ui.pushButton_15.clicked.connect(self.next_clicked_10)
    
    def home_clicked(self):
        if self.ui.stackedWidget.currentIndex()>3 and self.ui.stackedWidget.currentIndex()<=6:
            reply = QMessageBox.information(self, 'Warn', 'The model has not been saved, do you want to save the model?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self.saveModel_clicked()
                self.ui.stackedWidget.setCurrentIndex(0)
            elif reply == QMessageBox.No:
                self.ui.stackedWidget.setCurrentIndex(0)
        else:
            self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.label_title_bar_top.setText("")
    
    #page0 
    def newModel_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        
    def importModel_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(10)
    
    #page1
    def open_clicked(self):
        filepath, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.filename = os.path.basename(filepath)
        print(filepath, filetype)
        self.ui.lineEdit_2.setText(filepath)
        
    def next_clicked_1(self):
        self.df_name = self.ui.lineEdit_2.text()
        self.ui.stackedWidget.setCurrentIndex(2)

    #page2
    def regression_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(7)
    
    def parameterOptimization_clicked(self):
        global op
        op = Surrogate_model()
        self.ui.stackedWidget.setCurrentIndex(3)
    
    #page3
    def RBF_clicked(self):
        self.gotoWaitingPage('Initial Building...')
        op.init_build(self.df_name)
        self.ui.label_title_bar_top.setText(f"model_1 - {self.filename}")
        self.ui.stackedWidget.setCurrentIndex(4)
    
    def KRG_clicked(self):
        self.gotoWaitingPage('Initial Building...')
        op.init_build(self.df_name)
        self.ui.stackedWidget.setCurrentIndex(4)
    
    def KPLS_clicked(self):
        self.gotoWaitingPage('Initial Building...')
        op.init_build(self.df_name)
        self.ui.stackedWidget.setCurrentIndex(4)
    
    #page4    
    def next_clicked_4(self):
        input_list = [0]*12
        input_list[0] = self.ui.lineEdit_7.text()
        input_list[1] = self.ui.lineEdit_8.text()
        input_list[2] = self.ui.lineEdit_10.text()
        input_list[3] = self.ui.lineEdit_11.text()
        input_list[4] = self.ui.lineEdit_29.text()
        input_list[5] = self.ui.lineEdit_28.text()
        input_list[6] = self.ui.lineEdit_12.text()
        input_list[7] = self.ui.lineEdit_13.text()
        input_list[8] = self.ui.lineEdit_15.text()
        input_list[9] = self.ui.lineEdit_17.text()
        input_list[10] = self.ui.lineEdit_14.text()
        input_list[11] = self.ui.lineEdit_18.text()
        input_list = list(np.float_(input_list))
        
        global isOverBound
        isOverBound = False
        # bound check
        def boundCheck(index, name, lower, upper):
            if input_list[index]<lower or input_list[index]>upper: #注水比例
                QMessageBox.information(None, 'Over Bound Error', f'{name}請輸入{lower}~{upper}')
                global isOverBound
                isOverBound = True
                
        boundCheck(0, "電磁攪拌時間", 0, 180)
        boundCheck(1, "超音波震盪時間", 0, 360)
        boundCheck(2, "幫浦輸送速度", 0, 5)
        boundCheck(3, "沉浸液總體積", 0.05, 10)
        boundCheck(4, "沉浸液於散熱區體積", 0.05, 20)
        boundCheck(5, "冷卻瓦數", 0, 3)
        boundCheck(6, "電池直徑", 6, 21)
        boundCheck(7, "電池高度", 30, 70)
        boundCheck(8, "加熱區軸向熱傳導係數", 0.5, 400)
        boundCheck(9, "加熱區縱向熱傳導係數", 0.5, 400)
        boundCheck(10, "加熱瓦數", 0, 3)
        boundCheck(11, "環境溫度", 20, 40)
        if not isOverBound:        
            self.gotoWaitingPage('Training...')
            output0, output1, output2 = op.calculate(input_list)
            self.ui.label_21.setText(f"Water injection ratio: {round(output0,4)} wt%")
            self.ui.label_22.setText(f"Graphene concentration: {round(output1,4)} wt%")
            self.ui.label_20.setText(f"Alumina Concentration: {round(output2,4)} wt%")
            self.ui.label_49.setVisible(False)
            self.ui.lineEdit_3.setVisible(False)
            self.ui.stackedWidget.setCurrentIndex(5)
            
        
    
    #page5
    def next_clicked_5(self):
        if not self.isShortedFlag:
            if self.ui.comboBox_18.currentIndex() == 1:# not shorted
                self.ui.label_49.setVisible(True)
                self.ui.lineEdit_3.setVisible(True)
                self.isShortedFlag = True
            else:# is shorted
                self.gotoWaitingPage('Training...')
                op.retrain(75)
                op.model.train()
                self.ui.stackedWidget.setCurrentIndex(6)
        else:   
            self.gotoWaitingPage('Training...')
            op.retrain(float(self.ui.lineEdit_3.text()))
            op.model.train()
            self.isShortedFlag = False
            self.ui.stackedWidget.setCurrentIndex(6)
        
    
    #page6
    def continue_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(4)

    def saveModel_clicked(self):
        filepath, filetype = QFileDialog.getSaveFileName(self, 'Save File',"model_1")
        if filepath:
            filename = os.path.basename(filepath)
            filedir = os.path.dirname(filepath)
            new_filepath = filedir+f"/{filename}"
            print("----------------"+new_filepath)
            try:
                os.makedirs(new_filepath, exist_ok = True)
            except:
                os.remove(new_filepath)
                os.makedirs(new_filepath, exist_ok = True)
            op.save_model(f"{new_filepath}/{filename}.pkl")
            
            np.savetxt(f"{new_filepath}/X_train.csv", op.X_train, delimiter =",",fmt ='% s')
            np.savetxt(f"{new_filepath}/y_train.csv", op.y_train, delimiter =",",fmt ='% s')
            
            zf = zipfile.ZipFile('{}.zip'.format(filename), 'w', zipfile.ZIP_DEFLATED)
        
            for root, dirs, files in os.walk(filename):
                for file_name in files:
                    zf.write(os.path.join(root, file_name))
            
            shutil.rmtree(new_filepath)
            
            QMessageBox.information(None, 'Message', "保存完成")
            self.ui.label_title_bar_top.setText("")
            self.ui.stackedWidget.setCurrentIndex(0)
    #page7
    def Lasso_clicked(self):
        self.gotoWaitingPage('Training...')
        self.showRegressionResult('Lasso')
        self.ui.stackedWidget.setCurrentIndex(8)

    def Ridge_clicked(self):
        self.gotoWaitingPage('Training...')
        self.showRegressionResult('Ridge')
        self.ui.stackedWidget.setCurrentIndex(8)
    
    #page8
    def next_clicked_8(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    
    def showRegressionResult(self, algo):
        if algo == 'Lasso':
            result=regressType('Lasso',self.df_name)
        else:
            result=regressType('Ridge',self.df_name)
        self.ui.label_32.setText('train: '+str(round(result['(MSE) train'],4)))
        self.ui.label_33.setText(f'test: '+str(round(result['(MSE) test '],4)))
        self.ui.label_42.setText(f'train: '+str(round(result['(R^2) train'],4)))
        self.ui.label_43.setText(f'test: '+str(round(result['(R^2) test '],4)))
        self.showChart()

    def showChart(self):
        img = cv2.imread('compare.png')
        height, width, channel = img.shape
        bytesPerline = 3 * width
        qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.label.setPixmap(QPixmap.fromImage(qimg)) 
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda:self.update_frame(img))
        self.timer.start(100)

    def update_frame(self, frame):
        height = self.ui.label.height()
        width = int(frame.shape[1] * (height / frame.shape[0]))
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        h, w, c = frame.shape  # 获取图片形状
        image = QImage(frame, w, h, 3 * w, QImage.Format_RGB888)
        pix_map = QPixmap.fromImage(image)

        self.ui.label.setPixmap(pix_map)
        
    
    def gotoWaitingPage(self,label):
        self.ui.stackedWidget.setCurrentIndex(9)
        self.ui.label_3.setText(label)
        QApplication.processEvents()

    #page10
    def open_clicked_10(self):
        self.filepath, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.ui.lineEdit_4.setText(self.filepath)
        
    def next_clicked_10(self):
        self.reload_model()
        self.ui.stackedWidget.setCurrentIndex(4)
    
    def reload_model(self):
        with zipfile.ZipFile(self.filepath, 'r') as zf:
            zf.extractall(path='')
        dirname = os.path.dirname(self.filepath)
        purename = os.path.basename(self.filepath).split('.')[0]
        filepath = os.path.join(dirname, purename)
        global op
        op = Surrogate_model()
        op.model = op.load_model(f'{filepath}/{os.path.basename(purename)}.pkl')
        op.X_train = genfromtxt(f'{filepath}/X_train.csv', delimiter=',', skip_header = 1)
        op.y_train = genfromtxt(f'{filepath}/y_train.csv', delimiter=',', skip_header = 1)
        shutil.rmtree(filepath)
        self.ui.label_title_bar_top.setText(purename)
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    window = MainWindow()
    
    window.show()
    
    sys.exit(app.exec_())
    
# pyuic5 GUI_BASE.ui -o UI.py