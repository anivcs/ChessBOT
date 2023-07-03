# ChessBOT
To Run Code:
  Make sure you're in the latest version of gym-chess and are using gym version 17
  Change every instance of np.int -> np.int_ in board encoding file (from your gym-chess package)
  In envs.py (from your gym-chess package):
    Add info to return section of reset function
    Add info and truncated to return section of step function

To fix the 'ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.':
  convert self._observation() into an array from the envs.py file (from your gym-chess package)
