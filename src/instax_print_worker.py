import threading
from queue import Queue

if __package__:
    from .job import PrintJob
else:  # python src/main.py で直接実行する場合
    from job import PrintJob


class InstaxPrintWorker(threading.Thread):
    """印刷要求を順番に受け取るInstax用ワーカースレッド。"""

    def __init__(self, print_queue: Queue[PrintJob | None]) -> None:
        super().__init__(name="instax-print-worker")
        self.print_queue = print_queue

    def run(self) -> None:
        while True:
            job = self.print_queue.get()
            try:
                if job is None:
                    return
                self.print_image(job)
            finally:
                self.print_queue.task_done()

    def print_image(self, job: PrintJob) -> None:
        """後からInstax接続・印刷処理を実装するための差し替えポイント。"""
        print("Instax印刷要求を受信しました（印刷処理は未実装）")
        print(f"  image_path: {job.image_path}")
        print(f"  created_at: {job.created_at.isoformat(timespec='milliseconds')}")
        print(f"  person_ids: {job.person_ids}")
