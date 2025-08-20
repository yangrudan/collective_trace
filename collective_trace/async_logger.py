"""
Async logger for logging in a separate thread
"""

import queue
import time
import threading
from typing import Optional


class AsyncLogger:
    """Async logger that logs messages in a separate thread"""

    def __init__(self, log_file: Optional[str] = None, max_queue_size: int = 10000):
        """
        Init method for AsyncLogger class

        Args:
            log_file: If not None, the file to write logs to.
            max_queue_size: Maximum number of messages to keep in the queue before blocking.
        """
        self.log_file = log_file
        self.log_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._log_worker, daemon=True)
        self._worker_thread.start()

    def _log_worker(self) -> None:
        """Worker method for logging in a separate thread"""
        while not self._stop_event.is_set():
            try:
                message = self.log_queue.get(
                    timeout=1
                )  # Balance between block and exitable
                if self.log_file:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(message + "\n")
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except ValueError as e:
                print(f"Error in async logger worker thread: {str(e)}")

    def log(
        self, message: str, block: bool = True, timeout: Optional[float] = None
    ) -> bool:
        """
        Commit a log message to the queue

        Args:
            message: commit message
            block: whether to block until the message is committed
            timeout: block timeout in seconds

        Returns:
            bool: whether the message was successfully committed
        """
        try:
            self.log_queue.put(message, block=block, timeout=timeout)
            return True
        except queue.Full:
            print(f"Log queue is full, message : {message[:50]}...")
            return False

    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Flush the log queue, waiting for all messages to be committed

        Args:
            timeout: block timeout in seconds
        """
        start_time = time.perf_counter()
        while True:
            if self.log_queue.empty() and self.log_queue.unfinished_tasks == 0:
                return

            if timeout is not None and (time.perf_counter() - start_time) >= timeout:
                raise TimeoutError(
                    f"Logger flush timeout ({timeout}s）， \
                                   log queue size: {self.log_queue.qsize()}"
                )

            time.sleep(0.1)

    def close(self) -> None:
        """Close the logger"""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self.flush()


class RankedAsyncLogger(AsyncLogger):
    """Automatically append rank info when logging"""

    def __init__(
        self, base_log_file: Optional[str], rank: int, max_queue_size: int = 10000
    ):
        """
        Initialize a RankedAsyncLogger object

        Args:
            base_log_file: log file name
            rank: rank of current process
            max_queue_size: (default: 10000)
        """
        ranked_log_file = f"{base_log_file}-{rank}" if base_log_file else None
        super().__init__(log_file=ranked_log_file, max_queue_size=max_queue_size)
        self.rank = rank
