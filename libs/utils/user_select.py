import os
import re
from simple_term_menu import TerminalMenu

def user_select(pattern, dir_path):
    files = [f for f in [fs for _, _, fs in os.walk(dir_path)]]
    files = [f for f in files[0] if re.search(pattern, f)]
    files.sort()

    terminal_menu = TerminalMenu(files)
    choice_index = terminal_menu.show()

    return files[choice_index]
