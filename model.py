from brian2 import *


def get_oddball_seq(occurrences):
    #                     n occurrences of i
    seq = np.concatenate([np.ones(n, int)*i for i, n in enumerate(occurrences)])
    np.random.shuffle(seq)  # seq now describes the oddball sequence (e.g. with A=0, B=1, O=2)
    return seq


def get_stream_kernels(stream_indices, kernel, N_streams):
    '''
    Fills a size (len(stream_indices)+1, N_streams) array with input kernel values. The final row is left empty (zeros) to represent omissions.
    Args:
        stream_indices: List of receptive field centers (0, ..., N_streams-1). Order as in @fn:get_oddball_seq's `occurrences` parameter.
            E.g., if occurrences are [10, 4, 2] for [A, B, O], and A is stream #1 and B is stream #3, then stream_indices should be [1, 3].
        kernel: List of input values, starting from the stream center.
        N_streams: Number of streams.
    Example:
        > get_stream_kernels([1,3], [1,0.5], 5)
        < array([[0.5, 1. , 0.5, 0. , 0. ],
                 [0. , 0. , 0.5, 1. , 0.5],
                 [0. , 0. , 0. , 0. , 0. ]])
    '''
    kernels = np.zeros((len(stream_indices) + 1, N_streams))
    for i, idx in enumerate(stream_indices):
        for j, k in enumerate(kernel):
            for sign in (+1, -1):
                try:
                    kernels[i, idx + sign*j] = k
                except IndexError:
                    pass
    return kernels


def get_oddball_inputs(occurrences, stream_indices, params):#kernel=stim_kernel, N_streams=N_streams, timing=stim_timing):
    seq = get_oddball_seq(occurrences)
    kernels = get_stream_kernels(stream_indices, params['stim_kernel'], params['N_streams'])
    array = []
    for stream in seq:
        for _ in range(params['stim_timing'][0]):  # Stimulus on, apply stim kernel
            array.append(kernels[stream])
        for _ in range(params['stim_timing'][1]):  # Stimulus off, apply omission kernel (= zeros)
            array.append(kernels[-1])
    return seq, np.stack(array)


def make_neurons(Net, equations, params, state_monitors=['P', 'S', 'circuit'], spike_monitors=['P', 'S', 'circuit']):
    n = params['N_streams']
    circuit_neurons = NeuronGroup(4*n, equations['circuit'], threshold='v > v_th', reset='v = v0', refractory=params['tau_ref'], name='circuit', method='exact', namespace=params)
    Net.add(circuit_neurons)

    PEP = circuit_neurons[:n]
    PEN = circuit_neurons[n:2*n]
    IP = circuit_neurons[2*n:3*n]
    IN = circuit_neurons[3*n:]

    P = NeuronGroup(n, equations['P'], threshold='v > v_th', reset='v = v0', refractory=params['tau_ref'], name='P', method='euler', namespace=params)
    P.w_pred = 0
    Net.add(P)

    S = NeuronGroup(n, equations['S'], threshold='v > v_th', reset='v = v0', refractory=params['tau_ref'], name='S', method='euler', namespace=params)
    Net.add(S)

    neurons = {'PEP': PEP, 'PEN': PEN, 'IP': IP, 'IN': IN, 'P': P, 'S': S, 'circuit': circuit_neurons}

    states = {label: StateMonitor(group,
                                  ['v', 'w_pred'] if label == 'P' else ['v'],
                                  record=True)
              for label, group in neurons.items()
              if label in state_monitors}
    Net.add(states)

    spikes = {label: SpikeMonitor(group, record=True)
              for label, group in neurons.items()
              if label in spike_monitors}
    Net.add(spikes)

    return neurons, states, spikes


def make_column_synapses(Net, neurons, params):
    synapses = {
        'S_IN': Synapses(neurons['S'], neurons['IN'], on_pre='I_exc += w_S_IN', delay=params['delay_S_IN'], name='S_IN', namespace=params),
        'S_PEP': Synapses(neurons['S'], neurons['PEP'], on_pre='I_exc += w_S_PEP', delay=params['delay_S_PEP'], name='S_PEP', namespace=params),
        'P_IP': Synapses(neurons['P'], neurons['IP'], on_pre='I_exc += w_P_IP', delay=params['delay_P_IP'], name='P_IP', namespace=params),
        'P_PEN': Synapses(neurons['P'], neurons['PEN'], on_pre='I_exc += w_P_PEN', delay=params['delay_P_PEN'], name='P_PEN', namespace=params),
        'IP_PEP': Synapses(neurons['IP'], neurons['PEP'], on_pre='I_inh += w_IP_PEP', delay=params['delay_IP_PEP'], name='IP_PEP', namespace=params),
        'IN_PEN': Synapses(neurons['IN'], neurons['PEN'], on_pre='I_inh += w_IN_PEN', delay=params['delay_IN_PEN'], name='IN_PEN', namespace=params),

        'PEP_P': Synapses(neurons['PEP'], neurons['P'], on_pre='w_pred = clip(w_pred + pred_learning_rate_PEP, 0, 1)', name='PEP_P', namespace=params),
        'PEN_P': Synapses(neurons['PEN'], neurons['P'], on_pre='w_pred = clip(w_pred - pred_learning_rate_PEN, 0, 1)', name='PEN_P', namespace=params),
    }
    Net.add(synapses)

    for syn in synapses.values():
        syn.connect(j='i')

    return synapses


def make_lateral_synapses(Net, neurons, params):
    Lateral = Synapses(neurons['PEP'], neurons['PEN'], on_pre='I_exc += lateral_weight_scale/(i-j)**2', name='Lateral', namespace=params)
    Lateral.connect(condition='i != j')
    Lateral.delay = 'lateral_delay_scale * abs(i-j)'
    Net.add(Lateral)

    return {'Lateral': Lateral}
