import os
import sys

# Añadir el directorio del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from game.pages.home import HockeyMainScreen

# Ejecutar el menú principal
if __name__ == "__main__":
    main_screen = HockeyMainScreen()
    result = main_screen.run()
    print(f"Resultado: {result}")