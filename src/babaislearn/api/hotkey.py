import os
import signal
import time

import pynput
from pynput.keyboard import Key, KeyCode

class Hotkey:
    """Keyboard input access."""
    stopped: int = 0
    training_started: int = 0
    pressed: int = 1
    keyboard = pynput.keyboard.Controller()

    # Keys1
    UP1 = Key.up
    DOWN1 = Key.down
    LEFT1 = Key.left
    RIGHT1 = Key.right
    WAIT1 = Key.space
    UNDO1 = 'z'
    # Keys2
    UP2 = 'w'
    DOWN2 = 's'
    LEFT2 = 'a'
    RIGHT2 = 'd'
    WAIT2 = 'q'
    UNDO2 = 'x'

    @staticmethod
    def press_space() -> None:
        Hotkey.keyboard.press(Key.space)
        time.sleep(0.1)
        Hotkey.keyboard.release(Key.space)

    @staticmethod
    def _hotkeys(key: Key | str) -> None:
        if isinstance(key, Key) and key == Key.f10:
            os.kill(os.getpid(), signal.SIGTERM)
            print("--Program terminated--")
        elif isinstance(key, Key) and key == Key.f8:
            Hotkey.stopped = 1
            print("--Sent a stop signal to stop current training loop--")
        elif isinstance(key, KeyCode) and key.char == '+':
            Hotkey.training_started = 1
            print("--Training loop started--")
            time.sleep(0.5)
            Hotkey.training_started = 0

    listener = pynput.keyboard.Listener(on_press=_hotkeys)
    listener.start()