STATES = ('S', 'E', 'I', 'M', 'O', 'H')
NUM_STATES = len(STATES)
NUM_TRANS = 7
COLORS = ['green', 'orange', 'red', 'pink', 'gray', 'blue']


class STATE:
    S = 0  # susceptible
    E = 1  # exposed without symptom
    I = 2  # infected and with symptom
    M = 3  # with medical care
    O = 4  # out of system, dead  or cured
    H = 5  # maximum number of beds in hospital


class TRANS:
    S2E = 0
    E2I = 1
    I2M = 2
    I2O = 3
    M2O = 4
    EbyE = 5
    EbyI = 6
