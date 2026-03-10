"""GUI Application - College Canteen Face Detection System

Thin launcher that creates a DI Container and delegates to the GUI app.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.logging_config import setup_logging
from app.container import Container
from app.ui.gui_app import CanteenFaceDetectionGUI


def main():
    setup_logging()
    container = Container()
    app = CanteenFaceDetectionGUI(container)
    app.run()


if __name__ == "__main__":
    main()
