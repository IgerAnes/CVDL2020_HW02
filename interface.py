from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, 
                            QWidget, QPushButton, QLabel, QComboBox, 
                            QVBoxLayout, QHBoxLayout, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSlot
import sys
from process_function import AppWindow

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        MW_Layout = QGridLayout() #set main window arrange mode
        MW_Layout.addWidget(self.Background_Subtraction_Groupbox(), 0, 0) # add widget to ui and set the widget position
        MW_Layout.addWidget(self.Optical_Flow_Groupbox(), 1, 0)
        MW_Layout.addWidget(self.Perspective_Transform_Groupbox(), 2, 0)
        MW_Layout.addWidget(self.PCA_Groupbox(), 3, 0)
        self.setLayout(MW_Layout)

    def Background_Subtraction_Groupbox(self):
        AW = AppWindow()
        BackgroundSubtractionGroupbox = QGroupBox("1. Background Subtraction")
        ExecuteButton = QPushButton("1.1 Background Subtraction", self)
        ExecuteButton.clicked.connect(lambda:AW.Background_Subtraction_Func())

        BSG_Layout = QGridLayout()
        BSG_Layout.addWidget(ExecuteButton, 0, 0)
        BackgroundSubtractionGroupbox.setLayout(BSG_Layout)
        return BackgroundSubtractionGroupbox

    def Optical_Flow_Groupbox(self):
        AW = AppWindow()
        OpticalFlowGroupbox = QGroupBox("2. Optical Flow")
        Execute0Button = QPushButton("2.1 Preprocessing", self)
        Execute1Button = QPushButton("2.2 Video Tracking", self)
        Execute0Button.clicked.connect(lambda:AW.Preprocessing_Func())
        Execute1Button.clicked.connect(lambda:AW.VideoTracking_Func())

        OFG_Layout = QGridLayout()
        OFG_Layout.addWidget(Execute0Button, 0, 0)
        OFG_Layout.addWidget(Execute1Button, 1, 0)
        OpticalFlowGroupbox.setLayout(OFG_Layout)
        return OpticalFlowGroupbox

    def Perspective_Transform_Groupbox(self):
        AW = AppWindow()
        PerspectiveTransformGroupbox = QGroupBox("3. Perspective Transform")
        ExecuteButton = QPushButton("3.1 Perspective Transform", self)
        # ExecuteButton.clicked.connect(lambda:AW.SIFT_Keypoint_Func())

        PTG_Layout = QGridLayout()
        PTG_Layout.addWidget(ExecuteButton, 0, 0)
        PerspectiveTransformGroupbox.setLayout(PTG_Layout)
        return PerspectiveTransformGroupbox

    def PCA_Groupbox(self):
        AW = AppWindow()
        PCAGroupbox = QGroupBox("4. PCA")
        Execute0Button = QPushButton("4.1 Image Reconstruction", self)
        Execute1Button = QPushButton("4.2 Compute the reconstruction error ", self)
        Execute0Button.clicked.connect(lambda:AW.Image_Reconstruction_Func())
        Execute1Button.clicked.connect(lambda:AW.Calculate_Reconstruction_Error_Func())

        PG_Layout = QGridLayout()
        PG_Layout.addWidget(Execute0Button, 0, 0)
        PG_Layout.addWidget(Execute1Button, 1, 0)
        PCAGroupbox.setLayout(PG_Layout)
        return PCAGroupbox


if __name__ == "__main__":
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())