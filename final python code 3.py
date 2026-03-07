import paho.mqtt.client as mqtt
import keyboard
import time
import threading

# MQTT Configuration
BROKER = "1887637a30c2432683efd36d80813f78.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "titaniumrobotics"
PASSWORD = "Sapt090059#"
TOPIC = "rccar/control"

# ========== SPEED CONFIGURATION ==========
# Speed levels 1-9 for FORWARD/BACKWARD only
SPEED_LEVELS = {
    1: 10,   # Level 1: Very slow (10%)
    2: 15,   # Level 2: Slow (15%)
    3: 20,   # Level 3: Gentle (20%)
    4: 25,   # Level 4: Easy (25%)
    5: 30,   # Level 5: Moderate (30%)
    6: 35,   # Level 6: Good pace (35%)
    7: 40,   # Level 7: Fast (40%)
    8: 45,   # Level 8: Very fast (45%)
    9: 50,   # Level 9: Maximum (50%)
}

current_speed_level = 5      # Start at level 5 (30%)
CONSTANT_STEERING = 25       # GENTLE STEERING (25% of max) - stays constant
MOVE_DURATION = 1.0          # How long each keypress moves the bot (seconds)
UPDATE_RATE = 0.02           # Command send rate (50 times per second)
# =========================================

# Control state
throttle = 0
steering = 0
move_timer = None           # Timer that stops the bot after 1 second
is_moving = False           # Is the bot currently in a timed move?
lock = threading.Lock()     # Thread safety for shared variables

mqtt_client = None


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to MQTT Broker! Return code: {reason_code}")
    if reason_code == 0:
        print("Successfully connected!\n")
        print("=" * 60)
        print("    RC CAR CONTROLS - SPEED LEVEL MODE")
        print("=" * 60)
        print("D        - Forward  (1 second at current speed level)")
        print("A        - Backward (1 second at current speed level)")
        print("S        - Left Turn (1 second - always 25%)")
        print("W        - Right Turn (1 second - always 25%)")
        print("")
        print("1-9      - Set Speed Level for D/A")
        print("           1 = 10% (slowest), 5 = 30%, 9 = 50% (fastest)")
        print("")
        print("SPACE    - Emergency Stop")
        print("ESC      - Quit")
        print("=" * 60)
        print(f"\nCurrent Settings:")
        print(f"  Speed Level    : {current_speed_level} → {SPEED_LEVELS[current_speed_level]}%")
        print(f"  Steering (W/S) : {CONSTANT_STEERING}% (fixed)")
        print(f"  Move Duration  : {MOVE_DURATION} second per keypress")
        print("\n💡 Press 1-9 to change speed, then use D/A to move!")
        print("=" * 60 + "\n")
    else:
        print(f"Failed to connect, return code {reason_code}")


def send_command(t, s):
    """Send throttle and steering values to the car."""
    if mqtt_client:
        message = f"{t},{s}"
        mqtt_client.publish(TOPIC, message, qos=1)

        direction = ""
        if t > 0 and s == 0:
            direction = "↑ FORWARD"
        elif t < 0 and s == 0:
            direction = "↓ BACKWARD"
        elif t > 0 and s > 0:
            direction = "↗ FORWARD-RIGHT"
        elif t > 0 and s < 0:
            direction = "↖ FORWARD-LEFT"
        elif t < 0 and s > 0:
            direction = "↘ BACKWARD-RIGHT"
        elif t < 0 and s < 0:
            direction = "↙ BACKWARD-LEFT"
        elif t == 0 and s > 0:
            direction = "→ RIGHT TURN"
        elif t == 0 and s < 0:
            direction = "← LEFT TURN"
        else:
            direction = "● STOPPED"

        speed_display = f"[Level {current_speed_level}]" if t != 0 else ""
        print(f"Throttle: {t:4}% | Steering: {s:4}% | {direction:20} {speed_display:12}", end='\r')


def stop_bot():
    """Called by timer after MOVE_DURATION — stops the bot."""
    global throttle, steering, is_moving
    with lock:
        throttle = 0
        steering = 0
        is_moving = False
    send_command(0, 0)


def trigger_move(new_throttle, new_steering):
    """
    Start a 1-second timed move.
    If a move is already running, cancel it and start fresh.
    """
    global throttle, steering, move_timer, is_moving

    with lock:
        # Cancel any existing timer
        if move_timer is not None:
            move_timer.cancel()
            move_timer = None

        throttle = new_throttle
        steering = new_steering
        is_moving = True

    # Send command immediately
    send_command(new_throttle, new_steering)

    # Schedule stop after MOVE_DURATION seconds
    with lock:
        move_timer = threading.Timer(MOVE_DURATION, stop_bot)
        move_timer.daemon = True
        move_timer.start()


def get_current_speed():
    """Get the current speed percentage based on selected level."""
    return SPEED_LEVELS[current_speed_level]


def main():
    global mqtt_client, current_speed_level

    # Setup MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.on_connect = on_connect
    mqtt_client = client

    print("Connecting to MQTT broker...")
    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    client.loop_start()
    time.sleep(2)

    print("RC Car Ready!")
    print(f"Current speed level: {current_speed_level} ({SPEED_LEVELS[current_speed_level]}%)")
    print("Press 1-9 to change speed, then D/A to move!\n")

    # Track which keys were already pressed (to detect NEW presses only)
    key_was_pressed = {
        'w': False,
        's': False,
        'a': False,
        'd': False,
        'space': False,
        '1': False, '2': False, '3': False, '4': False, '5': False,
        '6': False, '7': False, '8': False, '9': False,
    }

    try:
        while True:
            # ---- Speed level selection (1-9) ----
            for num in range(1, 10):
                num_key = str(num)
                if keyboard.is_pressed(num_key):
                    if not key_was_pressed[num_key]:
                        key_was_pressed[num_key] = True
                        current_speed_level = num
                        print()
                        print(f"🔧 Speed level changed to {current_speed_level} → {SPEED_LEVELS[current_speed_level]}%" + " " * 30)
                else:
                    key_was_pressed[num_key] = False

            # ---- FORWARD (uses current speed level) ----
            # Hardware issue: D key actually moves forward
            if keyboard.is_pressed('d'):
                if not key_was_pressed['d']:
                    key_was_pressed['d'] = True
                    speed = get_current_speed()
                    print()
                    print(f"→ D pressed: FORWARD at Level {current_speed_level} ({speed}%) for 1 second...")
                    trigger_move(speed, 0)
            else:
                key_was_pressed['d'] = False

            # ---- BACKWARD (uses current speed level) ----
            # Hardware issue: A key actually moves backward
            if keyboard.is_pressed('a'):
                if not key_was_pressed['a']:
                    key_was_pressed['a'] = True
                    speed = get_current_speed()
                    print()
                    print(f"→ A pressed: BACKWARD at Level {current_speed_level} ({speed}%) for 1 second...")
                    trigger_move(-speed, 0)
            else:
                key_was_pressed['a'] = False

            # ---- LEFT (fixed steering, no speed levels) ----
            # Hardware issue: S key actually turns left
            if keyboard.is_pressed('s'):
                if not key_was_pressed['s']:
                    key_was_pressed['s'] = True
                    print()
                    print(f"→ S pressed: LEFT TURN at {CONSTANT_STEERING}% for 1 second...")
                    trigger_move(0, -CONSTANT_STEERING)
            else:
                key_was_pressed['s'] = False

            # ---- RIGHT (fixed steering, no speed levels) ----
            # Hardware issue: W key actually turns right
            if keyboard.is_pressed('w'):
                if not key_was_pressed['w']:
                    key_was_pressed['w'] = True
                    print()
                    print(f"→ W pressed: RIGHT TURN at {CONSTANT_STEERING}% for 1 second...")
                    trigger_move(0, CONSTANT_STEERING)
            else:
                key_was_pressed['w'] = False

            # ---- EMERGENCY STOP ----
            if keyboard.is_pressed('space'):
                if not key_was_pressed['space']:
                    key_was_pressed['space'] = True
                    global move_timer, is_moving
                    with lock:
                        if move_timer:
                            move_timer.cancel()
                            move_timer = None
                        is_moving = False
                    send_command(0, 0)
                    print()
                    print("*** EMERGENCY STOP ***")
            else:
                key_was_pressed['space'] = False

            # ---- QUIT ----
            if keyboard.is_pressed('esc'):
                print("\n\nStopping car and exiting...")
                with lock:
                    if move_timer:
                        move_timer.cancel()
                break

            time.sleep(UPDATE_RATE)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        send_command(0, 0)
        time.sleep(0.5)
        client.loop_stop()
        client.disconnect()
        print("Disconnected safely.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ESP32 RC Car Controller - Speed Level Mode")
    print("=" * 60)
    print("\nHOW TO USE:")
    print("  1. Press a number key (1-9) to select your speed")
    print("  2. Press D to move FORWARD at that speed for 1 second")
    print("  3. Press A to move BACKWARD at that speed for 1 second")
    print("  4. Press W/S to turn (always fixed at 25%)")
    print("")
    print("EXAMPLES:")
    print("  • Press '1' then 'D' → moves forward slowly (10%)")
    print("  • Press '5' then 'D' → moves forward moderately (30%)")
    print("  • Press '9' then 'D' → moves forward fast (50%)")
    print("  • Press '3' then 'A' → moves backward gently (20%)")
    print("")
    print("Each movement lasts exactly 1 second.")
    print("=" * 60 + "\n")

    main()
