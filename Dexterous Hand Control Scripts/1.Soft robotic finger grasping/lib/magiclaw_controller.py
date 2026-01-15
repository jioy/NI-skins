#!/usr/bin/env python

"""
MagiClaw Controller
=================

This script is to send control commands to the MagiClaw system.

"""

import argparse
import time
import threading
from typing import Tuple
from lib.modules.zmq import ClawPublisher, ClawSubscriber


class MagiclawController:
    """
    MagiClaw Controller

    This class handles the communication with the MagiClaw system.
    It allows sending commands to set the claw angle and motor speed,
    and receiving data from the MagiClaw.

    Attributes:
        publisher (ClawPublisher): Publisher for sending commands.
        subscriber (ClawSubscriber): Subscriber for receiving data.
        send_angle (float): The angle to set the claw.
        send_motor_speed (float): The speed of the motor.
        receive_angle (float): The received claw angle.
        receive_motor_angle (float): The received motor angle.
        receive_motor_speed (float): The received motor speed.
        receive_motor_iq (float): The received motor IQ.
        receive_motor_temperature (int): The received motor temperature.
    """

    def __init__(
        self, controller_host: str, magiclaw_host: str, loop_rate: int = 100
    ) -> None:
        """
        Initialize the MagiClaw controller.

        Args:
            controller_host (str): The host address for the controller.
            magiclaw_host (str): The host address for the MagiClaw.
        """

        # Create a publisher and a subscriber
        self.publisher = ClawPublisher(host=controller_host, port=5301)
        self.subscriber = ClawSubscriber(host=magiclaw_host, port=5301, timeout=10)

        # Initialize command and received data
        self.send_angle = 60.0
        self.send_motor_speed = 1000
        self.receive_angle = None
        self.receive_motor_angle = None
        self.receive_motor_speed = None
        self.receive_motor_iq = None
        self.receive_motor_temperature = None

        # Start the run loop in a separate thread
        self.run_thread = threading.Thread(target=self.run, args=(loop_rate,))
        self.run_thread.daemon = True
        self.run_thread.start()

    def close(self) -> None:
        """
        Close the publisher and subscriber.
        """

        self.publisher.close()
        self.subscriber.close()
        print("Publisher and subscriber closed.")

    def send_commands(self, claw_angle: float, motor_speed: float) -> None:
        """
        Send commands to the MagiClaw.

        Args:
            claw_angle (float): The angle to set the claw.
            motor_speed (float): The speed of the motor.
        """

        self.send_angle = claw_angle
        self.send_motor_speed = motor_speed

    def receive_data(self) -> Tuple[float, float, float, float, int]:
        """
        Receive data from the subscriber.

        Returns:
            data (Tuple[float, float, float, float, int]):
                - Claw angle (float)
                - Motor angle (float)
                - Motor speed (float)
                - Motor IQ (float)
                - Motor temperature (int)
        """
        return (
            self.receive_angle,
            self.receive_motor_angle,
            self.receive_motor_speed,
            self.receive_motor_iq,
            self.receive_motor_temperature,
        )

    def run(self, loop_rate: int = 100) -> None:
        """
        Run the main loop for publishing and subscribing to messages.

        Args:
            loop_rate (int): The loop rate in Hz (default is 100).
        """

        # Calculate the loop interval
        if loop_rate <= 0:
            raise ValueError("Loop rate must be greater than 0.")
        loop_interval = 1.0 / loop_rate

        try:
            while True:
                # Record the start time for this iteration
                tick_start = time.time()
                
                # Publish command
                self.publisher.publishMessage(
                    claw_angle=self.send_angle,
                    motor_speed=self.send_motor_speed,
                )
                
                # Subscribe and receive data
                try:
                    (
                        claw_angle,
                        motor_angle,
                        _,
                        motor_speed,
                        motor_iq,
                        motor_temperature,
                    ) = self.subscriber.subscribeMessage()
                    self.receive_angle = claw_angle
                    self.receive_motor_angle = motor_angle
                    self.receive_motor_speed = motor_speed
                    self.receive_motor_iq = motor_iq
                    self.receive_motor_temperature = motor_temperature
                except Exception as e:
                    print(f"Error receiving data: {e}")

                # Calculate the time taken for this iteration
                tick_duration = time.time() - tick_start
                sleep_duration = max(0.0, loop_interval - tick_duration)
                time.sleep(sleep_duration)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.close()


if __name__ == "__main__":
    controller = MagiclawController(
        controller_host="0.0.0.0",
        magiclaw_host="192.168.31.120",
        loop_rate=100,
    )
    try:
        while True:
            controller.send_commands(claw_angle=60.0, motor_speed=1000)
            print(controller.receive_data())
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
