#!/usr/bin/env python

import re
import zmq
import pathlib
from typing import Tuple
from datetime import datetime
from lib.modules.protobuf import claw_msg_pb2


class ClawPublisher:
    """
    ClawPublisher class.

    This class is used to publish claw messages using ZeroMQ.

    Attributes:
        context (zmq.Context): The ZeroMQ context.
        publisher (zmq.Socket): The ZeroMQ publisher socket.
    """

    def __init__(
        self,
        host: str,
        port: int,
        hwm: int = 1,
        conflate: bool = True,
    ) -> None:
        """
        Publisher initialization.

        Args:
            host (str): The host address of the publisher.
            port (int): The port number of the publisher.
            hwm (int): High water mark for the publisher. Default is 1.
            conflate (bool): Whether to conflate messages. Default is True.
        """

        print("{:-^80}".format(" Claw Publisher Initialization "))
        print(f"Address: tcp://{host}:{port}")

        # Create a ZMQ context
        self.context = zmq.Context()
        # Create a ZMQ publisher
        self.publisher = self.context.socket(zmq.PUB)
        # Set high water mark
        self.publisher.set_hwm(hwm)
        # Set conflate
        self.publisher.setsockopt(zmq.CONFLATE, conflate)
        # Bind the address
        self.publisher.bind(f"tcp://{host}:{port}")

        # Read the protobuf definition for Claw message
        with open(
            pathlib.Path(__file__).parent.parent / "protobuf/claw_msg.proto",
        ) as f:
            lines = f.read()
        messages = re.search(r"message\s+Claw\s*{(.*?)}", lines, re.DOTALL)
        body = messages.group(1)
        print("Message Claw")
        print("{\n" + body + "\n}")

        print("Claw Publisher Initialization Done.")
        print("{:-^80}".format(""))

    def publishMessage(
        self,
        claw_angle: float = 0.0,
        motor_angle: float = 0.0,
        motor_angle_percent: float = 0.0,
        motor_speed: float = 0.0,
        motor_iq: float = 0.0,
        motor_temperature: int = 0,
    ) -> None:
        """
        Publish the message.

        Args:
            claw_angle (float): The angle of the claw. Default is 0.0.
            motor_angle (float): The angle of the motor. Default is 0.0.
            motor_angle_percent (float): The angle percent of the motor. Default is 0.0.
            motor_speed (float): The speed of the motor. Default is 0.0.
            motor_iq (float): The current of the motor in IQ format. Default is 0.0.
            motor_temperature (int): The temperature of the motor in Celsius. Default is 0.
        """

        # Set the message
        claw = claw_msg_pb2.Claw()
        claw.timestamp = datetime.now().timestamp()
        claw.angle = claw_angle
        claw.motor.angle = motor_angle
        claw.motor.angle_percent = motor_angle_percent
        claw.motor.speed = motor_speed
        claw.motor.iq = motor_iq
        claw.motor.temperature = motor_temperature

        # Publish the message
        self.publisher.send(claw.SerializeToString())

    def close(self):
        """
        Close ZMQ socket and context.
        """
        
        if hasattr(self, "publisher") and self.publisher:
            self.publisher.close()
        if hasattr(self, "context") and self.context:
            self.context.term()


class ClawSubscriber:
    """
    ClawSubscriber class.

    This class is used to subscribe to claw messages using ZeroMQ.

    Attributes:
        context (zmq.Context): The ZeroMQ context.
        subscriber (zmq.Socket): The ZeroMQ subscriber socket.
        poller (zmq.Poller): The ZeroMQ poller for the subscriber.
        timeout (int): Maximum time to wait for a message in milliseconds.
    """

    def __init__(
        self,
        host: str,
        port: int,
        hwm: int = 1,
        conflate: bool = True,
        timeout: int = 100,
    ) -> None:
        """
        Subscriber initialization.

        Args:
            host (str): The host address of the subscriber.
            port (int): The port number of the subscriber.
            hwm (int): High water mark for the subscriber. Default is 1.
            conflate (bool): Whether to conflate messages. Default is True.
            timeout (int): Maximum time to wait for a message in milliseconds. Default is 100 ms.
        """

        print("{:-^80}".format(" Claw Subscriber Initialization "))
        print(f"Address: tcp://{host}:{port}")

        # Create a ZMQ context
        self.context = zmq.Context()
        # Create a ZMQ subscriber
        self.subscriber = self.context.socket(zmq.SUB)
        # Set high water mark
        self.subscriber.set_hwm(hwm)
        # Set conflate
        self.subscriber.setsockopt(zmq.CONFLATE, conflate)
        # Connect the address
        self.subscriber.connect(f"tcp://{host}:{port}")
        # Subscribe the topic
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        # Set poller
        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)
        self.timeout = timeout

        # Read the protobuf definition for Claw message
        with open(
            pathlib.Path(__file__).parent.parent / "protobuf/claw_msg.proto",
        ) as f:
            lines = f.read()
        messages = re.search(r"message\s+Claw\s*{(.*?)}", lines, re.DOTALL)
        body = messages.group(1)
        print("Message Claw")
        print("{\n" + body + "\n}")

        print("Claw Subscriber Initialization Done.")
        print("{:-^80}".format(""))

    def subscribeMessage(self) -> Tuple[float, float, float, float, float, int]:
        """
        Subscribe the message.

        Returns:
            data (Tuple[float, float, float, float, float, int]):
                - angle (float): The angle of the claw.
                - motor_angle (float): The angle of the motor.
                - motor_angle_percent (float): The angle percent of the motor.
                - motor_speed (float): The speed of the motor.
                - motor_iq (float): The current of the motor in IQ format.
                - motor_temperature (int): The temperature of the motor in Celsius.

        Raises:
            RuntimeError: If no message is received within the timeout period.
        """

        # Receive the message
        if self.poller.poll(self.timeout):
            # Receive the message
            msg = self.subscriber.recv()

            # Parse the message
            claw = claw_msg_pb2.Claw()
            claw.ParseFromString(msg)
        else:
            raise RuntimeError("No message received within the timeout period.")
        return (
            claw.angle,
            claw.motor.angle,
            claw.motor.angle_percent,
            claw.motor.speed,
            claw.motor.iq,
            claw.motor.temperature,
        )

    def close(self):
        """
        Close ZMQ socket and context.
        """
        
        if hasattr(self, "subscriber") and self.subscriber:
            self.subscriber.close()
        if hasattr(self, "context") and self.context:
            self.context.term()
