"""Optional LRCLIB diagnostics must tolerate missing matches without changing alignment."""
import copy

import pytest

from mezmur.song import lrc_check


LINES = ["ሃሌሉያ", "ክብር ለስሙ", "አሜን", "ኢየሱስ ጌታ ነው"]


def cues():
    return [{"start": index * 5 + 1.5, "end": index * 5 + 3, "text": text}
            for index, text in enumerate(LINES)]


def candidate(lines=None, duration=30):
    lines = LINES if lines is None else lines
    return {"synced": "\n".join(f"[00:{index * 5 + 1:05.2f}] {text}" for index, text in enumerate(lines)),
            "lrclib_duration": duration, "url": "https://lrclib.net/api/get/test"}


def test_nonempty_timestamp_list_with_no_matching_text_returns_no_diagnostic():
    assert lrc_check(cues(), [candidate(["ሌላ ቃል"])], 30) is None


def test_repeated_cues_after_all_matching_marks_are_used_do_not_crash():
    repeated = [{"start": 1.5, "end": 2, "text": LINES[0]},
                {"start": 3, "end": 4, "text": LINES[0]}]
    assert lrc_check(repeated, [candidate([LINES[0]])], 30) is None


def test_unmatched_line_does_not_discard_other_valid_timing_comparisons():
    original = cues()
    extra = {"start": 2.5, "end": 3, "text": "ሌላ ቃል"}
    result = lrc_check([original[0], extra, *original[1:]], [candidate()], 30)
    assert result == {"source": candidate()["url"], "lines_compared": 4, "median_seconds": 0.5}


def test_later_candidate_is_checked_after_a_candidate_without_matching_lines():
    correct = candidate()
    result = lrc_check(cues(), [candidate(["ሌላ ቃል"]), correct], 30)
    assert result["source"] == correct["url"] and result["lines_compared"] == 4


@pytest.mark.parametrize("synced", [None, "", "[ar:Artist]\n[ti:Title]", "[00:01.00] !!!"])
def test_absent_or_unusable_timestamp_marks_return_no_diagnostic(synced):
    assert lrc_check(cues(), [candidate() | {"synced": synced}], 30) is None


def test_missing_candidates_empty_cues_and_wrong_duration_remain_optional():
    assert lrc_check(cues(), [], 30) is None
    assert lrc_check([], [candidate()], 30) is None
    assert lrc_check(cues(), [candidate(duration=40)], 30) is None


def test_too_few_or_distant_matches_do_not_become_timing_evidence():
    assert lrc_check(cues()[:3], [candidate()], 30) is None
    distant = [{**cue, "start": cue["start"] + 10} for cue in cues()]
    assert lrc_check(distant, [candidate()], 30) is None


def test_successful_diagnostic_preserves_input_cues_and_timestamps():
    original, candidates = cues(), [candidate()]
    before = copy.deepcopy((original, candidates))
    result = lrc_check(original, candidates, 30)
    assert result["lines_compared"] == 4 and result["median_seconds"] == 0.5
    assert (original, candidates) == before
