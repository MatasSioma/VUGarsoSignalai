import sys

from PySide6 import QtWidgets

from .gui.main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    if len(sys.argv) > 1:
        win.load(sys.argv[1])
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
