from google.colab import drive  # type: ignore


def mount_drive():
    drive.mount("/content/drive")
