import os
import subprocess
import time

from api.objects import OBJECTS_DICT

class BabaGame:
    """Run the game and access game data.
    
    Has two main tasks:
        - runs "Baba Is You" as a Python subprocess
        - implements API to access subprocess stdout pipe, get data, and process it during runtime
    
    To find and run game executable, os.chdir is used to change current path. Otherwise subprocess cannot properly open
    the game.
    """
    def __init__(self) -> None:
        self._gamepath: str = "C:\\Program files (x86)\\Steam\\steamapps\\common\\Baba Is You"
        self._autoflush: bool = True
        self._path: str = os.getcwd()
        self._game: subprocess.Popen[str]
        os.chdir(self._gamepath)

    def start_game(self) -> int:
        """Opens Baba Is You as a subprocess."""
        self._game = subprocess.Popen(['Baba Is You.exe'], 
                                          stdout=subprocess.PIPE,
                                          bufsize=0, 
                                          universal_newlines=True)
        os.chdir(self._path) # after opening the process, change back to original directory
        if self._game.poll() is None: 
            return 0
        else:
            return -1

    def flush_stdout(self) -> None:
        if self._game.poll() is None:
            self._game.stdout.flush()

    def set_autoflush(self, val: bool) -> None:
        """--Don't use this outside run.py--"""
        self._autoflush = val

    def autoflush(self) -> None:
        """--Don't use this outside run.py--"""
        while self._autoflush:
            self._game.stdout.readline()
            time.sleep(0.001)

    def _check_stdout(self) -> int:
        """Reads line from subprocess output and follow up with a command.
        
        Returns:
            -1 if game is no longer opened.  
            1 if string '------' is found in current line. This happens after a command is issues by bot.  
            2 if string '###' is found, signaling a victory.
        """
        while True:
            check_line = self._game.stdout.readline()
            if self._game.poll() is not None:
                return -1
            elif "------" in check_line:
                return 1
            elif "###" in check_line:
                return 2 # managed to win

    def _api_data(self) -> list[str]:
        """Read redirected output from Baba Is You

        Output includes level data (each object + its (x,y) location) + player status (whether something is currently 
        You or not)

        Returns:
            A list of strings which contain environment data. If game is closed, returns empty list.
        """
        api_data: list[str] = []
        while True:
            if self._game.poll() is not None:
                return [] # return empty list when game is no longer open
            line = self._game.stdout.readline().removesuffix('\n')
            if '------' in line or '###' in line: # must be used to break out of loop and update file contents
                break
            elif len(line) > 0:
                api_data.append(line)
        return api_data

    def _create_mapstate(self, obj_list: list[str]) -> list[list[int]]:
        """Create a 2d list representing a grid of current map environment, with each element matching to a tile.
        
        All tiles have numerical values: for actual objects, values are 1 and onward. For empty space, 0 is used. For
        unaccessable areas i.e. tiles outside current map size; value -1 is used instead.

        ///Partial state vs Full state///  
        Currently, state can only display a single object on each tile. This object is always the first object 
        internally i.e. if there are rock,10,12 and tile,10,12 in this order -> only rock,10,12 is included. 
        Importantly, with current partial state network, 'baba' is the only special object which takes priority over 
        any other object.

        A fully expanded state tensor would be very large: 33x18x|OBJECT_DICT|. Now, OBJECT_DICT has over 120 objects 
        listed and even with 33x18x120 = 71280, this size would be enormous compared to current 33x18 = 594. 
        In this tensor, each map tile would contain a list of length >120, and each element of this list would then be 
        a amount of each object. 

        E.g. if L(x,y) refer to map tile list at (x,y) and index 0 refers to 'baba', then 
        L(1,1)[0] = 1 if one 'baba' is at location (1,1),
        L(1,1)[0] = 2 if two 'baba',
        L(1,1)[0] = 0 if no 'baba' at that location, etc.

        Maybe a better implementation exists, but this would describe a given mapstate completely. Now, training this 
        network would be *extremelty* difficult.
        ///

        Args:
            obj_list (list[str]): List of all api strings from baba game. These don't include empty spaces, only actual 
            objects and their location coordinates.

        Returns:
            A 2d list of numerical values (list of lists, type int), each corresponding to some game object. If object 
            is not present in objects.OBJECTS_DICT, it gets either value 0 or -1.
        """
        border_val: int = OBJECTS_DICT["border"] # border object i.e. non-accessable tile
        state: list[list[int]] = [[border_val] * 33 for _ in range(18)]
        check: int = 0
        if obj_list:
            for y in range(18):
                for x in range(33):
                    for val in obj_list:
                        if ','+str(x+1)+','+str(y+1)+'+' in val:
                            get_val: int = OBJECTS_DICT[val[0:val.index(",")]]
                            state[y][x] = get_val # actual object
                            check = 1
                            break
                    if check == 0:
                        state[y][x] = OBJECTS_DICT["empty"] # empty space
                    check = 0
            for val in obj_list:
                # checks whether other objects share a tile with "baba". If so, "baba" takes priority.
                # if a full-state network will be implemented, this part must be removed.
                if 'baba,' in val and 'text_' not in val:
                    x_idx = val.index(",")+1
                    y_temp = val[x_idx:].index(",")
                    y_idx = x_idx+y_temp+1
                    x = int(val[x_idx: y_idx-1])-1
                    y = int(val[y_idx:val.index("+")])-1
                    state[y][x] = OBJECTS_DICT["baba"]
                    break
            return state
        else:
            return [[]]

    def get_state(self, debug_print: bool=False) -> tuple[list[list[int]], int]:
        """Returns game state, and identifier to check if state was terminal or not."""
        val = self._check_stdout()
        if val == -1:
            return [[]], -1
        raw_state = self._api_data()
        state = self._create_mapstate(raw_state)
        if debug_print:
            for obj in state:
                print(obj)
            print('###')
        return state, val