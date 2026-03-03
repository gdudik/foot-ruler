import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', required=True, help='where are the feet pics')
parser.add_argument('--output', '-o', required=True, help="where do we put them when we're done looking at them")
args = parser.parse_args()


WATCH_FOLDER = args.path

class NewFileHandler(FileSystemEventHandler):

    def on_created(self, event):
        # ignore directories
        if event.is_directory:
            return
        time.sleep(1)
        file_path = Path(event.src_path)
        filename = file_path.name

        print(f"New file detected: {filename}")
        
        board_num = filename[0:1]

        # ---- build command ----
        cmd = [
            sys.executable,
            "measure.py",
            "--image", str(file_path),
            "--board", board_num,
            "--path", args.output
        ]

        print("Running:", " ".join(cmd))

        subprocess.run(cmd)


def main():
    path = Path(WATCH_FOLDER)

    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, str(path), recursive=False)

    observer.start()
    print(f"Watching folder: {path}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()