STATES = ('S', 'E', 'I', 'M', 'H', 'O')
NUM_STATES = len(STATES)
COLORS = ['green', 'orange', 'red', 'pink', 'blue', 'gray']


class STATE:
    S = 0  # susceptible
    E = 1  # exposed without symptom
    I = 2  # infected and with symptom
    M = 3  # with medical care
    H = 4  # maximum number of beds in hospital
    O = 5  # out of system
