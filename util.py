#!/usr/bin/env python


class maybe_profile:
    def __call__(self, func):
        if hasattr(__builtins__, 'profile'):
            return profile(func)  # noqa
        return func
