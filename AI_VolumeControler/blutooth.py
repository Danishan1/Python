# import dbus

# def set_pulseaudio_volume(device_name, new_volume):
#     bus = dbus.SystemBus()
#     manager = dbus.Interface(bus.get_object("org.freedesktop.DBus", "/org/freedesktop/DBus"), "org.freedesktop.DBus.ObjectManager")

#     # Get all PulseAudio sinks (audio output devices)
#     sinks = [str(o) for o, _ in manager.GetManagedObjects().items() if "org.PulseAudio.Core1.Device" in _.keys()]

#     for sink_path in sinks:
#         device = dbus.Interface(bus.get_object("org.freedesktop.DBus", sink_path), "org.freedesktop.DBus.Properties")
#         props = device.GetAll("org.freedesktop.DBus.Properties.Device")

#         # Check if the device is a Bluetooth headphone and matches the name
#         if "Bluetooth" in props["Class"] and device_name in props["Name"]:
#             # Adjust the volume (0.0 to 1.0)
#             device.Volume = dbus.Double(new_volume, variant_level=1)
#             print(f"Volume set to {new_volume} for {props['Name']}")
#             break

# # Example: Set volume to 80% for a Bluetooth headphone named "MyBluetoothHeadphones"
# set_pulseaudio_volume("MyBluetoothHeadphones", 0.8)
