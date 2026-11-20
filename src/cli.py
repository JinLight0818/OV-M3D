"""Терминалын тусламжийг монгол хэлээр үзүүлнэ."""

import argparse


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(add_help=False, **kwargs)
        self._positionals.title = "Оролт"
        self._optionals.title = "Сонголтууд"
        self.add_argument("-h", "--help", action="help", help="энэ зааврыг харуулаад гарах")

    def format_help(self):
        return super().format_help().replace("usage: ", "Хэрэглээ  ", 1)

    def format_usage(self):
        return super().format_usage().replace("usage: ", "Хэрэглээ  ", 1)
