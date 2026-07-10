import torch #PyTorch library for tensor operations
import torch.nn as nn #neural network modules
import torch.nn.functional as F
import numpy as np

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with adaptation
        that inherits from PyTorch's nn.Module
    """
    
    def __init__(self, tau_m=20.0, v_rest=-65.0, v_th=-50.0, 
                 v_reset=-65.0, dt=1.0, r=1.0, tau_adapt=100.0):
        super().__init__()
        self.tau_m = tau_m # membrane time constant (how fast voltage leaks)
        self.v_rest = v_rest # resting potential (default -65 mV)
        self.v_th = v_th # spike threshold (default -50 mV)
        self.v_reset = v_reset # potential after spike
        self.dt = dt # simulation time step (1 ms)
        self.r = r # membrane resistance
        self.tau_adapt = tau_adapt # adaptation time constant (how fast threshold recovers)
        
    def forward(self, current, v_init=None, adapt_init=None):
        """ Simulate LIF neuron dynamics over time given input current."""
        time_steps = current.shape[0] # number of time steps in the input current
        batch_size = current.shape[1] if current.dim() > 1 else 1 # batch size (number of neurons)
        
        if v_init is None: # if no initial voltage provided, start at resting potential
            v = torch.full((batch_size,), self.v_rest) # initial voltage
        else:
            v = v_init # initial voltage provided by caller
            
        if adapt_init is None: # if no initial adaptation provided, start with no adaptation
            adapt = torch.zeros(batch_size) # initial adaptation (threshold increase)
        else:
            adapt = adapt_init # initial adaptation provided by caller
            
        spikes = [] # list to store spike outputs at each time step
        voltages = [] # list to store voltage traces at each time step
        
        # For each time step, get input current and calculate adaptive threshold (higher after spikes)
        for t in range(time_steps):
            I = current[t]
            
            # Adaptive threshold: increases after each spike
            v_th_effective = self.v_th + adapt
            
            # Membrane update
            dv = (-(v - self.v_rest) + self.r * I) / self.tau_m
            # Voltage change = (leak current + input current) divided by time constant
            v = v + dv * self.dt # Euler integration step
            
            # Spike detection
            spike = (v >= v_th_effective).float() # Returns 1 if voltage exceeds threshold, 0 otherwise
            
            # Reset
            v = v * (1 - spike) + self.v_reset * spike # If spike, reset voltage; otherwise keep it
             
            # Adaptation update: threshold increases after spike
            adapt = adapt * (1 - self.dt / self.tau_adapt) + spike * 10.0
            # Adaptation decays over time, but increases by 10 units for each spike

            spikes.append(spike)
            voltages.append(v.clone())
            
        # Stack all time steps into tensors, plus final adaptation state   
        return torch.stack(spikes), torch.stack(voltages), adapt 


class UnsupervisedSNN(nn.Module):
    """
    Fully unsupervised spiking neural network with:
    - STDP for weight learning
    - Lateral inhibition for competition
    - Winner-take-all classification
    - Homeostatic control of neural activity
    """
    
    def __init__(self, n_input, n_hidden, neuron_params=None, stdp_params=None, homeo_params=None):
        super().__init__()
        
        # Default neuron parameters if not provided
        if neuron_params is None:
            neuron_params = {
                'tau_m': 20.0, 
                'v_th': -50.0, 
                'v_reset': -65.0, 
                'v_rest': -65.0, 
                'dt': 1.0,
                'tau_adapt': 100.0
            }

        # Default STDP parameters if not provided    
        if stdp_params is None:
            stdp_params = {
                'a_plus': 0.1, # learning rate for potentiation
                'a_minus': 0.09, # learning rate for depression
                'tau_plus': 20.0, 
                'tau_minus': 20.0,
                'w_min': 0.0, # minimum synaptic weight
                'w_max': 1.0  # maximum synaptic weight
            }
                
        # One LIF neuron for each hidden unit
        self.n_hidden = n_hidden
        self.neurons = LIFNeuron(**neuron_params)
        
        # STDP weights (input → hidden)
        # Initialized randomly, will be updated by STDP learning rule. 
        # Where n_input is the number of input channels (e.g. 700 for SHD) 
        # and n_hidden is the number of neurons in the hidden layer (e.g. 100).
        self.weights = nn.Parameter(
            torch.randn(n_input, n_hidden) * 0.1,
            requires_grad=False # means no backpropagation
        )
    
        
        # Lateral inhibition matrix
        # Each neuron inhibits others, excites itself
        # torch.eye(n_hidden) = identity matrix (diagonal = 1), Multiply by 2.0 → self-excitation = 2.0
        # torch.ones(n_hidden, n_hidden) = matrix of all ones, Multiply by 0.5 → inhibition = -0.5 for all other neurons

        self.register_buffer('lateral', torch.eye(n_hidden) * 2.0 - torch.ones(n_hidden, n_hidden) * 0.5)
        
        # STDP rule
        self.a_plus = stdp_params['a_plus']
        self.a_minus = stdp_params['a_minus']
        self.tau_plus = stdp_params['tau_plus']
        self.tau_minus = stdp_params['tau_minus']
        self.w_min = stdp_params['w_min']
        self.w_max = stdp_params['w_max']
        
        # For tracking learning
        self.weight_history = []
        self.spike_history = []

        # Homeostasis parameters
        if homeo_params is None:
            homeo_params = {
                'target_rate': 10.0, # мэтавая частата (Гц)
                'tau_homeo': 5000.0,  # пастаянная часу (мс)
                'homeo_strength': 0.02, # сіла ўплыву
                'min_homeo_factor': 0.5,  # мінімальны множнік
                'max_homeo_factor': 2.0 # максімальны множнік
            }

        self.target_rate = homeo_params['target_rate']
        self.tau_homeo = homeo_params['tau_homeo']
        self.homeo_strength = homeo_params['homeo_strength']
        self.min_homeo_factor = homeo_params['min_homeo_factor']
        self.max_homeo_factor = homeo_params['max_homeo_factor']

        # For tracking firing rate of each neuron
        self.register_buffer('running_rate', torch.zeros(n_hidden))
        self.register_buffer('homeo_factor', torch.ones(n_hidden))

        # Base learning rates
        self.base_a_plus = self.a_plus
        self.base_a_minus = self.a_minus
            
    def stdp_update(self, pre_spikes, post_spikes, homeo_factor, dt=1.0):
        """STDP learning rule with homeostatic modulation"""
        
        # Пераканаемся, што dt — гэта тэнзар
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt, dtype=torch.float32, device=pre_spikes.device)
        
        time_steps = pre_spikes.shape[0]
        n_pre = pre_spikes.shape[1]
        n_post = post_spikes.shape[1]
        
        pre_trace = torch.zeros(n_pre, device=pre_spikes.device)
        post_trace = torch.zeros(n_post, device=post_spikes.device)
        delta_w = torch.zeros(n_pre, n_post, device=pre_spikes.device)
        
        decay_plus = torch.exp(-dt / self.tau_plus)
        decay_minus = torch.exp(-dt / self.tau_minus)
        
        for t in range(time_steps):
            pre_spike_t = pre_spikes[t]
            post_spike_t = post_spikes[t]
            
            pre_trace = pre_trace * decay_plus + pre_spike_t
            post_trace = post_trace * decay_minus + post_spike_t
            
            # Патэнцыяцыя
            pre_mask = pre_spike_t > 0
            if pre_mask.any():
                for i in torch.where(pre_mask)[0]:
                    delta_w[i, :] += self.a_plus * post_trace * homeo_factor
            
            # Дэпрэсія
            post_mask = post_spike_t > 0
            if post_mask.any():
                for j in torch.where(post_mask)[0]:
                    delta_w[:, j] -= self.a_minus * pre_trace / (homeo_factor[j] + 1e-8)
        
        new_weights = self.weights + delta_w * dt
        new_weights = torch.clamp(new_weights, self.w_min, self.w_max)
        return new_weights
    def update_homeostasis(self, spikes, dt=1.0):
        """
        Updates homeostatic factor based on spike rate
        Formula: running_rate = decay * running_rate + (1-decay) * instantaneous_rate
        """
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt, dtype=torch.float32, device=spikes.device)

        time_steps = spikes.shape[0]
        
        # Spike count per neuron
        spike_counts = spikes.sum(dim=0)
        
        # Instantaneous rate (spikes per second)
        instantaneous_rate = spike_counts / (time_steps * dt / 1000.0)
        
        # Exponential moving average (ν̃ᵢ)
        decay = torch.exp(-dt / self.tau_homeo)
        self.running_rate = decay * self.running_rate + (1 - decay) * instantaneous_rate
        
        # Calculate homeostatic factor
        # If rate > target -> factor < 1 (reduce learning)
        # If rate < target -> factor > 1 (increase learning)
        rate_error = self.target_rate - self.running_rate
        self.homeo_factor = 1.0 + self.homeo_strength * rate_error / (self.target_rate + 1e-8)
        self.homeo_factor = torch.clamp(self.homeo_factor, self.min_homeo_factor, self.max_homeo_factor)
        
    def forward(self, input_spikes, record=False):
        """
        Forward pass (Process input spikes through the network, 
        apply lateral inhibition, and update weights with STDP, and apply homeostasis)
        """
        time_steps = input_spikes.shape[0] # number of time steps in the input spike train
        n_input = input_spikes.shape[1] # number of input channels (should match self.weights.shape[0])
        
        # Compute input currents
        currents = torch.zeros(time_steps, self.n_hidden) # initialize current tensor for each time step and hidden neuron
        for t in range(time_steps): # for each time step, calculate the input current to hidden neurons by multiplying input spikes with synaptic weights
            currents[t] = input_spikes[t] @ self.weights # matrix multiplication: [n_input] @ [n_input, n_hidden] → [n_hidden]
        
        # Simulate with lateral inhibition
        hidden_spikes = [] # list to store hidden layer spikes at each time step
        voltages = [] # list to store voltage traces at each time step
        adaptations = [] # list to store adaptation traces at each time step
        
        v = torch.zeros(self.n_hidden) # initial voltage for hidden neurons
        adapt = torch.zeros(self.n_hidden) # initial adaptation for hidden neurons
        last_spikes = torch.zeros(self.n_hidden) # to keep track of which neurons spiked in the previous time step for lateral inhibition
        
        for t in range(time_steps):
            # Lateral inhibition: previous spikes inhibit other neurons
            inhibition = self.lateral @ last_spikes # Calculate inhibition based on which neurons spiked in the last time step
            current_t = currents[t] - inhibition # Apply inhibition to the input current for this time step
            
            # Update neurons
            spike, v, adapt = self._update_neuron(current_t, v, adapt) 
            
            hidden_spikes.append(spike) 
            voltages.append(v.clone())
            adaptations.append(adapt.clone())
            last_spikes = spike 
        
        hidden_spikes = torch.stack(hidden_spikes) # Convert list of spikes to tensor [time_steps, n_hidden]
        
        # STDP learning (online, after each step)
        with torch.no_grad(): 
            new_weights = self.stdp_update(input_spikes, hidden_spikes, self.homeo_factor) 
            self.weights.data = new_weights
            # Update homeostasis after processing the sample
            self.update_homeostasis(hidden_spikes, dt=self.neurons.dt)
            
        # Record for analysis
        if record and len(self.weight_history) < 100:
            self.weight_history.append(self.weights.clone())
            self.spike_history.append(hidden_spikes.clone())
        
        # Classification: which hidden neuron fired the most?
        spike_counts = hidden_spikes.sum(dim=0)
        winner = torch.argmax(spike_counts)
        
        return hidden_spikes, winner
    
    def _update_neuron(self, current, v, adapt):
        """Single step LIF update with adaptive threshold"""
        tau_m = self.neurons.tau_m
        v_rest = self.neurons.v_rest
        v_th = self.neurons.v_th
        v_reset = self.neurons.v_reset
        dt = self.neurons.dt
        tau_adapt = self.neurons.tau_adapt
        
        # Adaptive threshold
        v_th_effective = v_th + adapt
        
        # Membrane update
        dv = (-(v - v_rest) + current) / tau_m
        v = v + dv * dt
        
        # Spike detection
        spike = (v >= v_th_effective).float()
        
        # Reset
        v = v * (1 - spike) + v_reset * spike
        
        # Adaptation
        adapt = adapt * (1 - dt / tau_adapt) + spike * 10.0
        
        return spike, v, adapt