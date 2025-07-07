import api.baba_api as baba_api

def get_state(debug_print: bool=False) -> tuple[list[list[int]], int]:
    val = baba_api.check_stdout()
    if val == -1:
        return [[]], -1
    raw_state = baba_api.api_data()
    state = baba_api.create_mapstate(raw_state)
    if debug_print:
        for obj in state:
            print(obj)
        print('###')
    return state, val