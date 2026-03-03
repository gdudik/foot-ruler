import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_FOLDER = "/path/to/watch"

class NewFileHandler(FileSystemEventHandler):

    def on_created(self, event):
        # ignore directories
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        filename = file_path.name

        print(f"New file detected: {filename}")

        # ---- extract data from filename ----
        # example: meet_123_event_45.txt
        parts = filename.split("_")

        try:
            meet_id = parts[1]
            event_id = parts[3].split(".")[0]
        except Exception:
            print("Filename format not recognized")
            return

        # ---- build command ----
        cmd = [
            "python",
            "process_file.py",
            "--meet", meet_id,
            "--event", event_id,
            "--file", str(file_path)
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