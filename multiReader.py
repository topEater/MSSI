import os
import _thread

_thread.start_new_thread(os.system, ("python dataSampleV3.py 1",))
_thread.start_new_thread(os.system, ("python dataSampleV3.py 2",))
_thread.start_new_thread(os.system, ("python dataSampleV3.py 3",))
_thread.start_new_thread(os.system, ("python dataSampleV3.py 4",))