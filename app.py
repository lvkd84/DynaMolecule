import sys
from PyQt5.QtWidgets import QApplication
from src.view import *
from src.controller import *
from src.model import *

def main():
   app = QApplication(sys.argv)

   scr_size = app.primaryScreen().size()
   window = MainWindow(scr_size.height())
   window.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()