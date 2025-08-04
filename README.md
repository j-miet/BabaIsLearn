# BabaIsLearn

Reinforcement learning applied to puzzle video game **Baba Is You** using PyTorch.  

*Requires a copy of **Baba Is You** game*.

<u>Realistically</u>, this project only offers two applications:
- an api to access game map states
- implements a basic reinforce algorithm which can be used for learning purposes

It unfortunately cannot offer a scalable platform for training and implementing a working AI because

1. Game's official modding api is used for getting output data, because programming a simulator for this game would be
  quite a task, to put it lightly. For this reason, game must run at all times during training, and RL agent has to 
  follow game's input speed limitations, allowing it to realistically perform around *10 actions per second*. This is 
  a very low number, making training slow.

    Furthermore, current implementation of states is incomplete, as it can only display a single object per tile whereas
     in-game, multiple objects can be stacked. This could be expanded quite easily, but state data size would become 
     enormous; see the *Environment* section for more details.

2. More importantly, implementing a reward function seems not possible. Currently, only a victory gives it a sensible 
reward. This results into a policy where agent just runs around randomly until it wins, and only then it learns to 
update policy. And because of how complex the game is, implementing any intermediate rewards becomes extremely 
complicated. 

    Only reasonable way I could think of would require a pre-trained network with supervised learning (data pairs of 
    game state & player input), then use this pre-trained network as basis for policy network. Again, this would require
     so much more processing power and speed + a proper api...

TL;DR:
1. complete environment (minus some minor bugs) can be accessed, but updating states is *very slow* 
2. coming up with a reward function is an enormous task

## Environment

- game provides an official modding API, but it disables io/os libraries thus preventing direct output
- In order to get access game's environment data, following step are required
    1. run ``Baba is You`` as a Python subprocess
    2. use official modding api to add custom hook function which print all level objects as text after action inputs.
    3. redirect all print outputs into output pipe of subprocess
    4. process and save object data into a more accessable format (see below)
- to enable 2. step, ``Lua`` mod api folder and ``world_data.txt`` must be copied into level pack's folder; more on 
this later.

Game states are returned as 33x18 matrices, or more precisely as list of lists with 33 sublists, each containing 18 
values. For example, the initial state for first level of the base game, "Baba Is You", is given as

    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 37, 59, 0, 0, 0, 0, 0, 23, 37, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 5, 5, 5, 3, 5, 5, 5, 24, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 5, 5, 5, 3, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 37, 48, 0, 0, 0, 0, 0, 33, 37, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ]

Each number corresponds to a game object. Object values are found in ``src/babaislearn/api/objects.py``. There are 2 
values in particular which act as fillers: 0
 for empty tiles and -1 for inaccessable tiles (map borders if level size has less than 33 width/18 height).

One obvious problem is that 33x18 grid can only hold one object in each position. Pretty much all of game's levels have 
objects which can be stacked, but states will only include the first one they access internally. So with ``rock,12,10`` 
and ``tile,12,10``, both have same coordinate (12,10), but because rock is first in order, only it gets counted and rest
 are discarded.  
One solution would be to create a matrix where each element is yet another list keeping count of all objects on that 
tile. As states can have values from 1-124 (even more if custom words are added), state becomes a 33x18x120 tensor which
 has 73656 values instead of current 33x18=594. Now each coordinate (x,y) points to a list of length 124 where each list
element counts the objects. Order would matter: if ``baba`` is index 0 and no other objects exist in that tile, list 
becomes ``[1,0,0,...,0,0]`` whereas if ``brick`` corresponds to index 111, both ``baba`` and ``brick`` in the same tile 
yields ``[1,0,0,...,0,1,0,...,0,0]`` where second 1 is at index 111. This would then be repeated for all 594 tiles. But 
using inputs this large with limited training speed would be very bad, and at this point simulated environment becomes 
a must. 


## Installation

1. Find game installation folder and go to level packs folder called ``Worlds``. For Windows users with Steam, this is 
located in Program Files (x86) so the path is something like this:  
``C:\Program Files (x86)\Steam\steamapps\common\Baba Is You\Data\Worlds``
2. Copy any of the level packs and rename it. For example, to use base game, create a copy of ``baba`` folder and rename
 it something you recognize e.g. 'babaislearn'.
3. copy the contents of ``src\babaislearn\Game files`` folder and paste them into this level pack folder. Now 
'babaislearn' has ``Lua`` folder which contains the modding api stuff, and ``world_data.txt`` which simply enables 
modding for current level pack.

To verify level pack was installed correctly, run the game and select this map pack. It should be called 
'Baba Is Learn' and author is just '...'. You can change displayed name by editing ``world_data.txt``.


### Python

For Python packages, [**PyTorch**](https://pytorch.org/get-started/locally/) is used for reinforcement learning stuff. 
PyTorch has been tested on Python ``3.13``, but earlier versions should work fine.

Other third-party packages required:

    matplotlib==3.10.3
    pynput==1.7.8

These versions can be adjusted based on your Python version. However, for ``pynput`` 1.7.8 is recommended; I've faced 
some issues in the past with newer versions.



## Training

### Keyboard controls
As mentioned, training process happens via running the game so program must be able to simulate controls to get states, 
actions and rewards.

Set your game controls in ``src\babaislearn\api\hotkey.py``. Two sets of controls are used because this increases
 input speed and therefore training speed. The ``pynput`` library is used to 
 perform inputs: it requires the prefix ``Key`` for modifier keys, but otherwise letter, numbers and symbols can 
 be typed as strings. Default values for control layouts are:

    Up: Key.up (up arrow)
    Down: Key.down (downm arrow)
    Left: Key.left (left arrow)
    Right: Key.right (right arrow)
    Wait = Key.space (spacebar)
    Undo = 'z'

    Up = 'w'
    Down = 's'
    Left = 'a'
    Right = 'd'
    Wait = 'q'
    Undo = 'x'

*Optional* You can also change 'stop', 'exit' and 'start' hotkeys. Inside the ``_hotkeys`` method, there are three 
lines,

    if isinstance(key, Key) and key == Key.f10:  
  
    elif isinstance(key, Key) and key == Key.f8:

    elif isinstance(key, KeyCode) and key.char == '+':

First line is the 'exit' hotkey: this will instantly close program. It ends the training loop without saving + closes 
the game. Default key is ``Key.f10`` (F10).

Seconds line is 'stop' hotkey: this sends a signal to stop current training loop. After ongoing episode finishes, bot 
will stop the training loop, save model in file, and returns to UI. Game also stays open. Default key is ``Key.f8``(F8).

Third is for 'start' hotkey which is used to begin training loop for selected algorithm. Default key is ``+``.

### Select location of game executable

Open ``src/babaislearn/api/babagame.py`` then find the line

    self._path: str = "C:\\Program files (x86)\\Steam\\steamapps\\common\\Baba Is You"

Simply replace ``"C:\\Program files (x86)\\Steam\\steamapps\\common\\Baba Is You"`` with your game path.

### Begin training

Execute ``src\babaislearn\run.py``. This opens both Baba Is You as a Python subprocess, and the basic command line user 
interface. Here you can select any supported algorithm, they are selected via number input starting from 1.  
**Also, don't forget you can use ``Alt+Enter`` to switch into windowed mode**

Training must be done on each individual level: 
1. In-game, with your level pack selected, enter the overworld map and find any level tile. Go sit on top of that level,
 but don't enter it. 
2. Now, in your command line interface, select desired algorithm, type the number and press enter. This will count down 
to 0 and begin training loop. Some algorithms can have additional stuff like data display windows which open at this
point, you can ignore them.
3. Then press the starting hotkey (defaul value is ``+``) and algorithm enters the map and begins a training loop.

- In steps 1. and 2. order matters: if algorithm is selected and you move manually, the api begins to direct state 
output into pipe. This pipe gets blocked after a few steps, freezing the game. Then, if you proceed to step 3., the 
output gets desynced
- To change algorithm parameters, open ``run.py`` and edit the values there. Alternatively, search any algorithm in 
``src\babaislearn\algorithms`` to adjust internal behavior.

### Bugs
- sometimes input from ``pynput`` fails to register. This makes agent unable to get a new state, freezing the training 
process. Pressing any resumes training, but obviously requires human interraction.
    - to reduce input errors, you can search for ``self._input_delay`` variable in reinforce.py and increase it's value 
    to add time between inputs. Another variable ``self._input_delay2`` does the same, but for spacebar: it's used for 
    skipping level transition screens where a longer pause is recommended.

## Algorithms

### REINFORCE

Uses the standard loss function i.e. action log probability weighted rewards. By default, uses a baseline value network 
to help approximate the value function which should improve policy learning.