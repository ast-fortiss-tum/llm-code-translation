import logging
import os
import shutil
import socket
from pathlib import Path

logger = logging.getLogger(__name__)


def check_is_port_in_use(port: int) -> bool:
    """
    Checks whether the given local port is used.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def remove_file_or_directory(path: str | Path, raise_value_error=True):
    """param <path> could either be relative or absolute."""
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        # remove the directory and its contents
        shutil.rmtree(path)
    else:
        if raise_value_error:
            raise ValueError(f"path {path} is not a file or dir.")
        else:
            logger.warning(f"path {path} is not a file or dir.")


def remove_contents_of_directory(path: str | Path):
    """param <path> could either be relative or absolute."""
    if os.path.isdir(path):
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        raise ValueError(f"path {path} is not a directory.")


def copy_file(source: Path, destination: Path):
    """file copying with exception handling"""

    if os.path.isdir(destination):
        destination = destination / source.name
    try:
        shutil.copyfile(source, destination)

    except shutil.SameFileError:
        pass

    except IsADirectoryError as e:
        logger.debug(f"Destination is a directory: {e}")
        copy_file(source, destination / source.name)

    except PermissionError as pe:
        logger.exception(f"Permission denied: {pe}")
        raise pe

    except Exception as e:
        logger.error(f"Error occurred while copying file: {e}")
        raise e
