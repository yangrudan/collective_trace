"""
Detect if a function in collective operation is timed out
"""
import os
import time
import threading


class OperationTimer:
    """Independent timer class for operation timeout detection"""

    def __init__(self, timeout_threshold, callback):
        """
        Args:
            timeout_threshold: Timeout threshold in seconds
            callback: Timeout callback function with format: func(op_id, func_name, \
                      is_async, timed_out_type)
                      timed_out_type: "unfinished" (timed out without \
                      completion) / "finished_late" (completed but timed out)
        """
        self.timeout_threshold = timeout_threshold
        self.callback = callback  # Timeout callback
        # Operations to monitor: {op_id: (start_time, func_name, is_async,
        # is_completed)}
        self.pending_ops = {}
        self.lock = threading.Lock()  # Thread-safe lock for pending_ops
        self.monitor_thread = None
        self.running = False

    def start(self):
        """Start the monitoring thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

    def _monitor(self):
        """Periodically check if operations have timed out"""
        while self.running:
            current_time = time.perf_counter()
            with self.lock:
                # Iterate through all operations to monitor
                for op_id, (start_time, func_name, is_async, is_completed) in list(
                    self.pending_ops.items()
                ):
                    # Only check incomplete operations for timeouts
                    if not is_completed:
                        if current_time - start_time > self.timeout_threshold:
                            # Operation incomplete and timed out, trigger
                            # callback
                            self.callback(op_id, func_name, is_async, "unfinished")
                            # Mark as completed to avoid duplicate triggers
                            self.pending_ops[op_id] = (
                                start_time,
                                func_name,
                                is_async,
                                True,
                            )
            time.sleep(0.1)  # Reduce CPU usage

    def register_operation(self, op_id, func_name, is_async):
        """Register a new operation and start timing"""
        with self.lock:
            self.pending_ops[op_id] = (time.perf_counter(), func_name, is_async, False)

    def unregister_operation(self, op_id):
        """Delete a operation and timeout event"""
        if op_id in self.pending_ops:
            del self.pending_ops[op_id]

    def mark_completed(self, op_id):
        """Mark operation as completed and check if it finished late"""
        with self.lock:
            if op_id in self.pending_ops:
                start_time, func_name, is_async, _ = self.pending_ops[op_id]
                end_time = time.perf_counter()
                # Check if completed but timed out
                if end_time - start_time > self.timeout_threshold:
                    self.callback(op_id, func_name, is_async, "finished_late")
                # Remove completed operation
                del self.pending_ops[op_id]

    def is_timed_out(self, op_id):
        """Check if an operation has timed out"""
        with self.lock:
            if op_id not in self.pending_ops:
                return False
            start_time, _, _, is_completed = self.pending_ops[op_id]
            return not is_completed and (
                time.perf_counter() - start_time > self.timeout_threshold
            )

class TimeOutDaemon(OperationTimer):
    """
    Daemon thread class that monitors and handles timeouts for
    collective operations.

    This class retrieves the timeout threshold from environment
    variables or parameters,creates and manages a timer, and
    invokes the specified callback function when a timeout occurs.
    """

    # Define default timeout threshold as integer
    DEFAULT_TIMEOUT_THRESHOLD = 480  # 8 minutes (integer value)

    def __init__(self, callback):
        """Initialize a TimeOutDaemon instance"""
        timeout_threshold = self._resolve_timeout_threshold()
        super().__init__(timeout_threshold, callback)
        self.start()  # Start the monitoring thread in the constructor

    def _resolve_timeout_threshold(self):
        """Resolve the timeout threshold with priority: environment variable > default value"""
        env_timeout = os.environ.get("COLLECTIVE_TIMEOUT")
        if env_timeout is not None:
            try:
                return int(env_timeout)
            except ValueError:
                pass

        return self.DEFAULT_TIMEOUT_THRESHOLD

    def get_timeout_threshold(self):
        """Get the currently set timeout threshold in seconds"""
        return self.timeout_threshold
