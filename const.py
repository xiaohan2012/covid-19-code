STATES = ('S', 'E', 'I', 'M', 'O', 'H')
NUM_STATES = len(STATES)
COLORS = ['green', 'orange', 'red', 'pink', 'gray', 'blue']


class STATE:
    S = 0  # susceptible
    E = 1  # exposed without symptom
    I = 2  # infected and with symptom
    M = 3  # with medical care
    O = 4  # out of system, dead  or cured
    H = 5  # maximum number of beds in hospital
