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

    all_states = ('S', 'E', 'I', 'M', 'O', 'H')
    num_states = len(all_states)
    colors = ['green', 'orange', 'red', 'pink', 'gray', 'blue']
    state2color = {
        S: 'green',
        E: 'orange',
        I: 'red',
        M: 'pink',
        O: 'gray',
        H: 'blue'
    }
    

class TRANS:
    S2E = 0
    E2I = 1
    I2M = 2
    I2O = 3
    M2O = 4
    EbyE = 5
    EbyI = 6

    num_trans = 7


class STATE_VAC:
    S = 0  # susceptible
    E = 1  # exposed without symptom
    I = 2  # infected and with symptom
    M = 3  # with medical care
    O = 4  # out of system, dead  or cured
    
    V = 5  # vaccinated (but not taking effect yet)
    V1 = 6  # vaccinated but can transmit virus
    V2 = 7  # vaccinated and cannot transmit virus
    EV1 = 8  # V1 after becoming infected (without any symptom)

    H = 9  # maximum number of beds in hospital
    
    all_states = ('S', 'E', 'I', 'M', 'O') + ('V', 'V1', 'V2', 'EV1') + ('H', )
    num_states = len(all_states)

    colors = ['green', 'orange', 'red', 'pink', 'gray', 'blue']

    state2color = {
        S: 'green',
        E: 'orange',
        I: 'red',
        M: 'pink',
        O: 'gray',
        H: 'blue',
        V: 'violet',
        V1: 'magenta',
        V2: 'cyan',
        EV1: 'purple'
    }

class TRANS_VAC(TRANS):
    S2V = 7
    V2S = 8
    V_to_V1 = 9
    V_to_V2 = 10
    V1_to_EV1 = 11
    EV1_to_R = 12
    V2_to_R = 13

    num_trans = TRANS.num_trans + 7
