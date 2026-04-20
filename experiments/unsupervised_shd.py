import torch # PyTorch for tensor operations and neural network implementation
import tonic # Tonic library for event-based datasets and transformations
import tonic.transforms as transforms # For data transformations (not used in this script but commonly needed for event-based data)
import matplotlib.pyplot as plt # For visualizations of results and learned weights
import numpy as np # For numerical operations (not heavily used here but often useful for data manipulation)
import sys # To modify the system path for importing custom modules
sys.path.insert(0, 'D:/Sabina/Projects/snn_stdp') # Add project directory to path for importing custom network module
from network.unsupervised_snn import UnsupervisedSNN  # Import the custom unsupervised SNN class that implements STDP learning and lateral inhibition

def convert_events_to_spikes(events, time_steps=200, sensor_size=(700, 1, 1)):
    """Convert SHD events to spike tensor"""
    spikes = torch.zeros(time_steps, sensor_size[0]) # Initialize spike tensor with zeros (shape: [time_steps, n_channels])
    
    t_min = events['t'].min() # Get minimum timestamp to normalize time
    t_max = events['t'].max() # Get maximum timestamp to normalize time
     
    for event in events: # Iterate through each event and convert to spike tensor
        t_norm = (event['t'] - t_min) / (t_max - t_min) # Normalize timestamp to [0, 1]
        t_idx = min(int(t_norm * time_steps), time_steps - 1) # Convert normalized time to discrete time index, ensuring it does not exceed the maximum index
        channel = event['x'] # Get the channel index from the event (assuming 'x' contains the channel information)
        spikes[t_idx, channel] = 1.0 # Set the corresponding position in the spike tensor to 1 to indicate a spike at that time and channel
    
    return spikes

print("=" * 60)
print("UNSUPERVISED STDP CLASSIFIER")
print("No labels used during training!")
print("=" * 60)

# Load SHD dataset
print("\n1. Loading SHD dataset...")
train_dataset = tonic.datasets.SHD(save_to='./data', train=True) # Load training dataset (labels will be ignored during training)
test_dataset = tonic.datasets.SHD(save_to='./data', train=False) # Load test dataset (labels will be used only for evaluation after training)

print(f"   Train samples: {len(train_dataset)}") # Print the number of training samples loaded from the SHD dataset
print(f"   Test samples: {len(test_dataset)}") # Print the number of test samples loaded from the SHD dataset

# Create network
print("\n2. Creating unsupervised SNN with lateral inhibition...")
network = UnsupervisedSNN(
    n_input=700,
    n_hidden=200,
    neuron_params={'tau_m': 20.0, 'v_th': -50.0, 'dt': 1.0, 'tau_adapt': 50.0},
        stdp_params={
            'w_min': 0.0,           # Minimum weight
            'w_max': 1.0,           # Maximum weight
            'a_plus': 0.01,         # LTP learning rate
            'a_minus': 0.012,        # LTD learning rate
            'tau_plus': 20.0,       # LTP time constant (ms)
            'tau_minus': 20.0,      # LTD time constant (ms)
        }
    )

print(f"   Input: 700 channels → Hidden: 200 neurons")
print(f"   Learning: STDP + Lateral inhibition")
print(f"   Classification: Winner-take-all (no readout layer!)")

# Train on first 2000 samples (no labels!)
print("\n3. Unsupervised STDP training...")
print("   Presenting samples, no labels given...")

sample_count = 2000 # Limit to first 2000 samples for training to speed up the process (can be increased for better results)
neuron_responses = {i: [] for i in range(200)} # Dictionary to track which digit each hidden neuron responds to (for evaluation only, not used during training)

for idx in range(sample_count): # Iterate through the training samples (up to sample_count)
    events, label = train_dataset[idx] # Get the events and label for the current sample (label will be ignored during training)
    spikes = convert_events_to_spikes(events) # Convert the event data to a spike tensor that can be fed into the SNN (shape: [time_steps, n_channels])
    
    # Forward pass (STDP updates automatically)
    hidden_spikes, winner = network.forward(spikes)
    
    # Track which neuron responds to which digit (for evaluation only)
    neuron_responses[winner.item()].append(label)
    
    if (idx + 1) % 500 == 0: #` Print progress every 500 samples`
        print(f"   Processed {idx + 1}/{sample_count} samples")

print(f"\n   Training complete!")

# Map neurons to digits (winner-take-all)
print("\n4. Mapping neurons to digits (winner-take-all)...")
neuron_to_digit = {}

for neuron in range(200):
    responses = neuron_responses[neuron]
    if responses:
        # Most frequent digit this neuron responded to
        digit = max(set(responses), key=responses.count)
        neuron_to_digit[neuron] = digit

# Count how many neurons learned each digit
digit_counts = {d: 0 for d in range(20)}
for neuron, digit in neuron_to_digit.items():
    digit_counts[digit] += 1

print("   Neuron specialization:")
for digit in range(20):
    count = digit_counts[digit]
    print(f"   Digit {digit:2d} → {count} neurons")

# Test
print("\n5. Testing (still no labels!)...")
correct = 0
total = 0

test_samples = 1000
for idx in range(test_samples):
    events, true_label = test_dataset[idx]
    spikes = convert_events_to_spikes(events)
    
    _, winner = network.forward(spikes, record=False)
    
    predicted = neuron_to_digit.get(winner.item(), -1)
    
    if predicted == true_label:
        correct += 1
    total += 1
    
    if (idx + 1) % 200 == 0:
        print(f"   Tested {idx + 1}/{test_samples}")

accuracy = 100 * correct / total
print(f"\n{'='*60}")
print(f"UNSUPERVISED STDP CLASSIFICATION ACCURACY: {accuracy:.2f}%")
print(f"{'='*60}")
print(f"\nNo labels used during training!")
print(f"Network self-organized through STDP + lateral inhibition")

# Visualize learned weights
print("\n6. Visualizing learned weights...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Weight matrix
weights = network.weights.detach().numpy()
ax = axes[0, 0]
im = ax.imshow(weights, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
ax.set_xlabel('Hidden Neurons')
ax.set_ylabel('Input Channels')
ax.set_title('STDP-Learned Weights')
plt.colorbar(im, ax=ax)

# Weight distribution
ax = axes[0, 1]
ax.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel('Weight Value')
ax.set_ylabel('Count')
ax.set_title('Weight Distribution')

# Neuron specialization bar chart
ax = axes[1, 0]
digits = list(range(20))
counts = [digit_counts[d] for d in digits]
ax.bar(digits, counts)
ax.set_xlabel('Digit Class')
ax.set_ylabel('Number of Specialized Neurons')
ax.set_title('Neuron Specialization per Digit')

# Winning neuron activity
ax = axes[1, 1]
neuron_activity = [len(responses) for responses in neuron_responses.values()]
ax.hist(neuron_activity, bins=20, alpha=0.7, edgecolor='black')
ax.set_xlabel('Number of Samples Won')
ax.set_ylabel('Number of Neurons')
ax.set_title('Winner Neuron Distribution')

plt.tight_layout()
plt.savefig('unsupervised_stdp_results.png', dpi=150)
print("   Saved visualization to 'unsupervised_stdp_results.png'")
plt.show()