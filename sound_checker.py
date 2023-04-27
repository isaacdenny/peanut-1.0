import sounddevice as sd

# Get a list of available input devices
devices = sd.query_devices()

# Print the supported sample rates for each device
for device in devices:
    print(device['name'])
    print(device['max_input_channels'])
    print(device['default_samplerate'])

