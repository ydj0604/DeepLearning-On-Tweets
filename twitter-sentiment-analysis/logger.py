class Logger(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def write(self, msg, begin=False):
        if begin:
            with open(self.file_path, "w") as f:
                f.write(msg + '\n')
        else:
            with open(self.file_path, "a") as f:
                f.write(msg + '\n')
