import threading
import time

from api.babagame import BabaGame
from api.hotkey import Hotkey
import utils

# list of supported algorithms 
from algorithms.reinforce.reinforce import Reinforce

def run() -> None:
    babagame = BabaGame()
    babagame.start_game()
    print("\nType algorithm number to begin training, 'exit' to quit.\n"
                          "F8 to send an stop signal to current training loop.\n"
                          "F10 to shut down program (it's better to first press F8 and wait until policy is saved)\n"
                          "------------------------------\n"
                          ">List of supported algorithms:\n\n"
                          "1 -> REINFORCE")
    while True:
        babagame.set_autoflush(True)
        flush = threading.Thread(target=babagame.autoflush, daemon=True)
        flush.start()
        input_str = input("=> ")
        babagame.set_autoflush(False)
        match input_str:
            case '1':
                model = Reinforce(game=babagame,
                                  policy_lr=0.0001, 
                                  step_reward=-1, 
                                  failed_reward=-1, 
                                  success_reward=10,
                                  episodes=1000,
                                  steps=300)
                time.sleep(2)
                print(f"REINFORCE initialized, press '+' to begin.")
                while not Hotkey.training_started:
                    time.sleep(0.01)
                model.begin()
            case 'exit':
                break
            case 'test':
                utils.timer(2)
                while True:
                    if babagame.get_state(debug_print=True)[1] in {-1, 2}:
                        break

if __name__ == '__main__':
    run()