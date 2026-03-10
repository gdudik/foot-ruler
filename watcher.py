import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', required=True, help='Path to raw images. This folder will be watched for new images (required).')
parser.add_argument('--output', '-o', required=True, help="Location to output processed images (required).")
args = parser.parse_args()


WATCH_FOLDER = args.path

class NewFileHandler(FileSystemEventHandler):

    def on_created(self, event):
        # ignore directories
        if event.is_directory:
            return
        time.sleep(1)
        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in ['.jpg', '.jpeg']:
            return
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

        completed = subprocess.run(cmd, check=True, timeout=25)
        print(f"returned {completed.returncode}", flush=True)

        # try:
        #     result = subprocess.run(cmd, check=True, timeout= 25)
        # except subprocess.CalledProcessError as e:
        #     print(f"ERROR processing {filename}:")
        #     print(f"  Return code: {e.returncode}")
        #     print(f"  stdout: {e.stdout}")
        #     print(f"  stderr: {e.stderr}")
        # except subprocess.TimeoutExpired as e:
        #     print('timed out')
        # except Exception as e:
        #     print(f"UNEXPECTED ERROR processing {filename}: {e}")


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