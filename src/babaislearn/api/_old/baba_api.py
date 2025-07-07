"""API functions to return a enviromental state after each action + initial state.

Supports only partial states i.e. each tile can only include a single object (see below)

///Partial state vs Full state///  
Currently, state can only display a single object on each tile. This object is always the first object internally
i.e. if there are rock,10,12 and tile,10,12 in this order -> only rock,10,12 is included. Importantly, with current 
partial state network, 'baba' is the only special object which takes priority over any other object.

A fully expanded state tensor would be very large: 33x18x|OBJECT_DICT|. Now, OBJECT_DICT has over 120 objects 
listed and even with 33x18x120 = 71280, this size would be enormous compared to current 33x18 = 594. 
In this tensor, each map tile would contain a list of length >120, and each element of this list would then be a 
amount of each object. 

E.g. if L(x,y) refer to map tile list at (x,y) and index 0 refers to 'baba', then 
L(1,1)[0] = 1 if one 'baba' is at location (1,1),
L(1,1)[0] = 2 if two 'baba',
L(1,1)[0] = 0 if no 'baba' at that location, etc.

Maybe a better implementation exists, but this would describe a given mapstate completely. Now, training this network
would be *extremelty* difficult.
///
"""

import pathlib

from api.babagame import BabaGame
from api.objects import OBJECTS_DICT

class apiData:
    state: list[list[int]] = []

def check_stdout() -> int:
    """Reads line from subprocess output and follow up with a command.
    
    Returns:
        -1 if game is no longer opened.  
        1 if string '------' is found in current line. This happens after a command is issues by bot.  
        2 if string '###' is found, signaling a victory.
    """
    while True:
        check_line = BabaGame._game.stdout.readline()
        if BabaGame._game.poll() is not None:
            return -1
        elif "------" in check_line:
            return 1
        elif "###" in check_line:
            return 2 # managed to win

def api_data() -> list[str]:
    """Read redirected output from Baba Is You

    Output includes level data (each object + its (x,y) location) + player status (whether something is currently You 
    or not)

    Returns:
        A list of strings which contain environment data. If game is closed, returns empty list.
    """
    api_data: list[str] = []
    while True:
        if BabaGame._game.poll() is not None:
            return [] # return empty list when game is no longer open
        line = BabaGame._game.stdout.readline().removesuffix('\n')
        if '------' in line or '###' in line: # must be used to break out of loop and update file contents
            break
        elif len(line) > 0:
            api_data.append(line)
    return api_data

def create_mapstate(obj_list: list[str]) -> list[list[int]]:
    """Create a 2d list representing a grid of current map environment, with each element matching to a tile.
    
    All tiles have numerical values: for actual objects, values are 1 and onward. For empty space, 0 is used. For
    unaccessable areas i.e. tiles outside current map size; value -1 is used instead.

    Args:
        obj_list (list[str]): List of all api strings from baba game. These don't include empty spaces, only actual 
        objects and their location coordinates.

    Returns:
        A 2d list of numerical values (list of lists, type int), each corresponding to some game object. If object is
        not present in objects.OBJECTS_DICT, it gets either value 0 or -1.
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

# For testing/debugging only
def _check_stdout_test() -> int:
    """Reads line from subprocess output and follow up with a command.
    
    Returns:
        -1 if game is no longer opened.  
        1 if string '------' is found in current line. This happens after a command is issues by bot.  
        2 if bot managed to win, string '###' is found.
    """
    while True:
        check_line = BabaGame._game.stdout.readline()
        if BabaGame._game.poll() is not None:
            return -1
        elif "------" in check_line:
            with open(pathlib.Path(__file__).parent/'testdata.txt', 'w') as f:
                ...
            return 1
        elif "###" in check_line:
            return 2

def _api_data_test() -> int:
    """Read redirected output from Baba Is You and write it in a file.

    Output includes level data (each object + its (x,y) location) + player status (whether something is currently You 
    or not)

    Returns:
        -1 if game instance no longer exists.  
        1 otherwise.
    """
    with open(pathlib.Path(__file__).parent/'testdata.txt', 'a') as file:
        while True:
            if BabaGame._game.poll() is not None:
                return -1
            line = BabaGame._game.stdout.readline()
            print(line, end='')
            if '------' in line or '###' in line:
                break
            file.write(line)
    return 1