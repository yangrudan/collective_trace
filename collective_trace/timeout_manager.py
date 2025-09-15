"""
This module manages operation timeouts using a daemon process.
"""
import os
import signal
from .timeout_daemon import TimeOutDaemon

class TimeoutManager:
    def __init__(self, logger, timeout_callback):
        self.logger = logger
        self.timeout_daemon = TimeOutDaemon(timeout_callback)

    def register_operation(self, op_id, func_name, is_async):
        self.timeout_daemon.register_operation(op_id, func_name, is_async)

    def mark_completed(self, op_id):
        self.timeout_daemon.mark_completed(op_id)

    def is_timed_out(self, op_id):
        return self.timeout_daemon.is_timed_out(op_id)

    def get_timeout_threshold(self):
        return self.timeout_daemon.get_timeout_threshold()

    def unregister_operation(self, op_id):
        self.timeout_daemon.unregister_operation(op_id)

    def stop(self):
        self.timeout_daemon.stop()
        