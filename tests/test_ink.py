import numpy as np

from mezmur.ink import Flourish, InkTransitions


def test_flourishes_draw_and_finish():
    for seed in range(30):
        ev = Flourish(0.0, 1920, 1080, 560, seed)
        drawn = []
        for t in np.linspace(0, ev.duration, 12):
            m = np.zeros((1080, 1920), np.uint8)
            ev.draw(m, t)
            drawn.append(int((m > 0).sum()))
        assert max(drawn) > 0
        assert drawn[-1] <= max(drawn) * 0.2  # mostly gone by the end


def test_one_transition_per_line_change_and_frame_unchanged_elsewhere():
    ink = InkTransitions(1280, 720, [5.0, 5.4, 9.0, 14.0], (140, 214, 255), seed=1)
    assert len(ink.events) == 3  # 5.4 is too close to 5.0
    frame = np.full((720, 1280, 3), 40, np.uint8)
    assert (ink.apply(frame.copy(), 2.0) == frame).all()
    assert (ink.apply(frame.copy(), 5.0) != frame).any()
