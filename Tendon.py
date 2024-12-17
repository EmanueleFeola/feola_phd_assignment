class Tendon(object):
    LEN = 0.25  # m, tendon length. assumed constant.
    STIFFNESS = -1  # assume infinitely stiff tendon.

    def __init__(self, abs_root_path_):
        pass

    @staticmethod
    def get_len():
        return Tendon.LEN

    @staticmethod
    def get_stiffness():
        return Tendon.STIFFNESS
