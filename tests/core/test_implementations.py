import threading

import pytest

from gridzero.core import implementation
from gridzero.core.implementations import fast


def test_fast_paths_are_used_unless_the_reference_is_chosen():
    assert fast()
    with implementation("reference"):
        assert not fast()
    for impl in ("auto", "numba"):
        with implementation(impl):
            assert fast()


def test_a_block_restores_the_setting_around_it_also_after_an_error():
    with implementation("reference"):
        with implementation("numba"):
            assert fast()
        assert not fast()
        with pytest.raises(KeyError), implementation("numba"):
            raise KeyError("inside")
        assert not fast()
    assert fast()


def test_a_block_leaves_other_threads_on_their_own_setting():
    # The thread starts before the block: whether new threads copy the starting thread's
    # context depends on the Python build (free-threaded 3.14 does).
    ready, chosen, seen = threading.Event(), threading.Event(), []

    def other():
        ready.set()
        chosen.wait()
        seen.append(fast())

    thread = threading.Thread(target=other)
    thread.start()
    ready.wait()
    with implementation("reference"):
        chosen.set()
        thread.join()
        assert not fast()
    assert seen == [True]


@pytest.mark.parametrize(
    ("impl", "error"), [("scipy", ValueError), ("", ValueError), (None, TypeError), (1, TypeError)]
)
def test_unknown_implementations_are_rejected(impl, error):
    with pytest.raises(error), implementation(impl):
        pass
