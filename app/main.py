import sys
from PyQt5.QtWidgets import QApplication
from view import *
from controller import *
from model import *

def main():
   app = QApplication(sys.argv)

   scr_size = app.primaryScreen().size()
   window = MainWindow(scr_size.height())
   window.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()