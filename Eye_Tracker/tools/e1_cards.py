"""e1_cards.py -- E1 卡片序列的单一事实源(与 docs/E1_CARDS.md 一致,预注册不许改)。

口播名约定:网球 L/M/R 念"一号球/二号球/三号球"——斜位站上视觉左右会翻转,
数字名不带方位暗示,认球本身;打分仍用地图名 网球L/M/R。
"""

SPOKEN = {"网球L": "一号球", "网球M": "二号球", "网球R": "三号球",
          "苹果": "苹果", "石榴": "石榴", "香蕉": "香蕉"}
TOKEN = {"L": "网球L", "M": "网球M", "R": "网球R"}


def expand(seq_str):
    return [TOKEN.get(t, t) for t in seq_str.replace(",", " ").split()]


CARDS = {
    # 综合卡(6 物 ×12 项,seed 815)
    "e1": ("综合1(1m 标)", expand("苹果 M 石榴 香蕉 R L M 石榴 香蕉 L 苹果 R")),
    "e2": ("综合2(3m 标)", expand("L R 苹果 石榴 M 香蕉 苹果 石榴 香蕉 R L M")),
    "e3": ("综合3(4m 标)", expand("香蕉 L 苹果 M 石榴 R 香蕉 石榴 M 苹果 R L")),
    "e4": ("综合4(对角,能站多远站多远)", expand("苹果 L 香蕉 石榴 M R M 石榴 R L 香蕉 苹果")),
    "e5": ("综合5(3m 边慢走边盯)", expand("R 石榴 L 香蕉 苹果 M 香蕉 M L R 石榴 苹果")),
    # 纯球卡(seed 816/817/818)
    "s1": ("球卡S1(1m 标)", expand("R L M L M M R L L M R M R R L")),
    "s2": ("球卡S2(2m 标)", expand("M R M R L R M L R R M L M L L")),
    "s3": ("球卡S3(3m 标)", expand("M L L M R R L L M R L M R M R")),
    "s4": ("球卡S4(4m 标)", expand("L L M R R M M L R M R R M L L")),
    "s5": ("球卡S5(4m 斜位A α30° θ2.05°)", expand("M L M L M R R M R L L M L R R")),
    "s6": ("球卡S6(4m 斜位B α18° θ1.27°)", expand("L M R R M R R M L M L M L R L")),
    "s7": ("球卡S7(4m 斜位C α15° 贴遮挡极限 θ1.05°)", expand("L L R R M L M R R M L M R M L")),
}
