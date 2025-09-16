"""
This module manages operation timeouts using a daemon process.
"""

from .timeout_daemon import TimeOutDaemon


class TimeoutManager:
    """
    This class manages operation timeouts using a daemon process."""

    def __init__(self, logger, timeout_callback):
        """Initialize the timeout manager."""
        self.logger = logger
        self.timeout_daemon = TimeOutDaemon(timeout_callback)

    def register_operation(self, op_id, func_name, is_async):
        """Register an operation with the timeout manager."""
        self.timeout_daemon.register_operation(op_id, func_name, is_async)

    def mark_completed(self, op_id):
        """Mark an operation as completed."""
        self.timeout_daemon.mark_completed(op_id)

    def is_timed_out(self, op_id):
        """Check if an operation has timed out."""
        return self.timeout_daemon.is_timed_out(op_id)

    def get_timeout_threshold(self):
        """Get the timeout threshold."""
        return self.timeout_daemon.get_timeout_threshold()

    def unregister_operation(self, op_id):
        """Unregister an operation from the timeout manager."""
        self.timeout_daemon.unregister_operation(op_id)

    def stop(self):
        """Stop the timeout manager."""
        self.timeout_daemon.stop()
