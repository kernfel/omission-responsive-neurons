from brian2 import *


# Equations
equations = {
    'circuit': '''
        dv/dt = ((v0 - v) + R_m * (I_exc - I_inh)) / tau_m : volt (unless refractory)
        dI_exc/dt = -I_exc / tau_exc : ampere
        dI_inh/dt = -I_inh / tau_inh : ampere
    ''',

    'P': '''
        dv/dt = ((v0 - v) + R_m * w_pred*Pred(t)) / tau_m : volt (unless refractory)
        w_pred : 1
    ''',

    'S': '''
        dv/dt = ((v0 - v) + R_m * Stim(t, i)) / tau_m : volt (unless refractory)
    '''
}


params = {
# Neurons
    'R_m': 1*Mohm,
    'C_m': 30*nF,

    'v0': 0*mV,
    'v_th': 10*mV,
    'tau_ref': 2*ms,


# Synapses
    'tau_exc': 5*ms,
    'tau_inh': 20*ms,

    'w_S_IN': 0.5*uA,
    'w_S_PEP': 0.5*uA,
    'w_P_IP': 0.5*uA,
    'w_P_PEN': 0.5*uA,
    'w_IP_PEP': 0.2*uA,
    'w_IN_PEN': 0.2*uA,

    'delay_S_IN': 10*ms,
    'delay_S_PEP': 20*ms,
    'delay_P_IP': 10*ms,
    'delay_P_PEN': 20*ms,
    'delay_IP_PEP': 5*ms,
    'delay_IN_PEN': 5*ms,

    'pred_learning_rate_PEP': 0.001,
    'pred_learning_rate_PEN': 0.0003,

    'lateral_weight_scale': 1.2*uA,
    'lateral_delay_scale': 5*ms,


# Stimulation
    'N_streams': 5,
    'stim_timing': [2, 13],  # [0]*stim_dt on, [1]*stim_dt off
    'stim_dt': 10*ms,
    'input_strength': 0.2*uA,
    'stim_kernel': [1, 0.5]
}

params['tau_m'] = params['R_m'] * params['C_m']
