import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.py'):  # Adjust this to your specific file type if needed
            print(f'File changed: {event.src_path}. Running script...')
            subprocess.run(['python', 'all.py'])  # Replace with your script name

if __name__ == "__main__":
    path = '.'  # Directory to watch (current directory)
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()