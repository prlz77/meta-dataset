import argparse
parser = argparse.ArgumentParser()
argparse.parser = parser

class FLAGS(object):
    def update(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

argparse.FLAGS = FLAGS()

parse_args = parser.parse_args
def _parse_args(*args, **kwargs):
    argparse.FLAGS.update(vars(parse_args(*args, **kwargs)))
    return argparse.FLAGS

parser.parse_args = _parse_args
