import donkeycar as dk
import RPi.GPIO as GPIO

class TB6612Throttle:
    def __init__(self, pwm_controller, direction_channel):
        self.direction_channel = direction_channel
        self.pwm_controller = pwm_controller
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.direction_channel, GPIO.OUT)
        
    def run(self, throttle):
        pulse = dk.utils.map_range(abs(throttle), 0, 1, 0, 4095)
        if throttle > 0:
            GPIO.output(self.direction_channel, False)
        else:
            GPIO.output(self.direction_channel, True)
        self.pwm_controller.set_pulse(pulse)
        
    def shutdown(self):
        self.run(0)
