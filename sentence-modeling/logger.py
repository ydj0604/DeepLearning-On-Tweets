class Logger(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def write(self, msg):
        with open(self.file_path, "a") as f:
            f.write(msg + '\n')
