"""Define logger and its format."""

import logging
import time
import random
import string
import sys
import pathlib
logging.getLogger('matplotlib.font_manager').disabled = True

for _noisy in ('httpx', 'httpcore', 'openai', 'anthropic', 'boto3',
               'botocore', 'urllib3', 'google', 'google_genai', 'paramiko'):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

LOG_DIR = pathlib.Path(__file__).parent.parent.parent.resolve() / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

format="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s"

def logfilename(tag=""):
    """Construct a unique log file name: autocomp-date[-tag]-random.log"""
    timeline = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())
    randname = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    if tag:
        return f"autocomp-{timeline}-{tag}-{randname}.log"
    return f"autocomp-{timeline}-{randname}.log"

def move_log(log_dir, tag=""):
    logger.propagate = False
    formatter = logging.Formatter(format)
    logfile = log_dir / logfilename(tag)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    # Root logger has level NOTSET so all logs are passed to handlers,
    # which may or may not pass them through.
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

""" Initialize logger with file and console handlers. """
logfile = LOG_DIR / logfilename()
file_handler = logging.FileHandler(logfile)
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)

# Root logger has level NOTSET so all logs are passed to handlers,
# which may or may not pass them through.
logging.basicConfig(format=format, handlers=[file_handler, console_handler], level=logging.NOTSET)
logger = logging.getLogger(__name__)
