# !/usr/bin/env python3

"""
MagicLaw ZMQ Module
====================

This module provides ZeroMQ-based communication for the MagicLaw system, including publishers and subscribers
for various components such as claw, and MagiClaw.
"""

from .claw import ClawPublisher, ClawSubscriber