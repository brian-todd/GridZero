import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from game_demo import CLAIM, FLIP, HIDDEN, Claims, RandomClaimer
from gridzero.agents.llm import (
    LlamaServer,
    LLMPlayer,
    Message,
    Option,
    parse_choice,
)
from gridzero.game import Action, play, replay

GAME = Claims(length=5)


class FakeChat:
    """A model that answers with a fixed choice, or a seeded random one, and keeps every request."""

    def __init__(self, choice=None, answer=None):
        self.choice, self.answer, self.requests = choice, answer, []

    def __call__(self, messages, schema, seed):
        self.requests.append((messages, schema, seed))
        if self.answer is not None:
            return self.answer
        n = len(schema["properties"]["choice"]["enum"])
        choice = self.choice if self.choice is not None else np.random.default_rng(seed).integers(n)
        return json.dumps({"reason": "because", "choice": int(choice)})


@dataclass(frozen=True)
class ClaimsMenu:
    """`Claims` for a language model: the strip as it is shown, and each free cell twice."""

    def options(self, game, view, legal, /):
        return [
            Option(Action(verb, int(cell)), f"{name} cell {cell}")
            for verb, name in ((CLAIM, "claim"), (FLIP, "flip a coin for"))
            for cell in np.flatnonzero(legal[verb])
        ]

    def messages(self, game, view, options, /):
        strip = "".join({-1: ".", HIDDEN: "?"}.get(int(o), str(int(o))) for o in view)
        lines = [f"Strip: {strip}", *(f"{i}. {o.text}" for i, o in enumerate(options))]
        return [
            {"role": "system", "content": "Claim more cells than the other player."},
            {"role": "user", "content": "\n".join(lines)},
        ]


def middle():
    """A position of `Claims` with cell 1 claimed by a flip, so hidden, and cell 3 outright."""
    position = GAME.step(GAME.start(0), (FLIP, 1))
    return GAME.step(position, (CLAIM, 3))


def test_the_player_plays_the_option_it_is_told():
    position = middle()
    view, legal = GAME.view(position, 0), GAME.legal(position)
    options = ClaimsMenu().options(GAME, view, legal)
    for choice in (0, len(options) - 1):
        player = LLMPlayer(ClaimsMenu(), FakeChat(choice=choice))
        assert player(GAME, 0, view, legal, np.random.default_rng(0)) == options[choice].action


def test_the_model_is_shown_the_view_not_the_position():
    position = middle()
    chat = FakeChat(choice=0)
    LLMPlayer(ClaimsMenu(), chat)(
        GAME, 0, GAME.view(position, 0), GAME.legal(position), np.random.default_rng(0)
    )
    [(messages, _, _)] = chat.requests
    assert "Strip: .?.1." in messages[1]["content"]


def test_a_single_option_is_taken_without_asking():
    position = middle()
    legal = np.zeros_like(GAME.legal(position))
    legal[FLIP, 4] = True
    chat = FakeChat()
    action = LLMPlayer(ClaimsMenu(), chat)(
        GAME, 0, GAME.view(position, 0), legal, np.random.default_rng(0)
    )
    assert action == (FLIP, 4)
    assert chat.requests == []


def test_requests_carry_a_schema_of_the_menu_and_a_seed_from_the_generator():
    position = middle()
    view, legal = GAME.view(position, 0), GAME.legal(position)
    n = len(ClaimsMenu().options(GAME, view, legal))
    for chars in (50, 0):
        chat = FakeChat(choice=0)
        LLMPlayer(ClaimsMenu(), chat, reason_chars=chars)(
            GAME, 0, view, legal, np.random.default_rng(7)
        )
        [(messages, schema, seed)] = chat.requests
        assert schema["properties"]["choice"]["enum"] == list(range(n))
        assert schema["required"] == (["reason", "choice"] if chars else ["choice"])
        assert schema["additionalProperties"] is False
        if chars:
            assert schema["properties"]["reason"]["maxLength"] == chars
        else:
            assert "reason" not in schema["properties"]
        assert seed == np.random.default_rng(7).integers(2**31)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert f"{n - 1}. " in messages[1]["content"]  # every option is on the menu


def test_a_bad_answer_from_the_model_raises():
    position = middle()
    player = LLMPlayer(ClaimsMenu(), FakeChat(answer='{"choice": 999}'))
    with pytest.raises(ValueError, match="not on the menu"):
        player(GAME, 0, GAME.view(position, 0), GAME.legal(position), np.random.default_rng(0))


def test_games_against_a_model_end_and_replay():
    def game(seed):
        players = [LLMPlayer(ClaimsMenu(), FakeChat()), RandomClaimer()]
        return play(GAME, players, GAME.start(seed))

    first, again = game(4), game(4)
    assert first.final.over
    assert first.actions == again.actions
    assert list(replay(GAME, first))[-1].key() == first.final.key()


def test_parameters_are_checked(not_an_integer):
    with pytest.raises(ValueError, match="reason_chars"):
        LLMPlayer(ClaimsMenu(), FakeChat(), reason_chars=-1)
    with pytest.raises(TypeError, match="reason_chars"):
        LLMPlayer(ClaimsMenu(), FakeChat(), reason_chars=not_an_integer)
    with pytest.raises(ValueError, match="top_p"):
        LlamaServer(top_p=1.5)
    with pytest.raises(ValueError, match="timeout"):
        LlamaServer(timeout=0.0)
    with pytest.raises(TypeError, match="thinking"):
        LlamaServer(thinking=1)  # ty: ignore[invalid-argument-type]  # an int, on purpose


@pytest.mark.parametrize(
    "answer",
    [
        '{"choice": 9}',
        '{"choice": -1}',
        '{"choice": true}',
        '{"choice": 1.0}',
        '{"reason": "x"}',
        "3",
        "not json",
        "[1]",
    ],
)
def test_answers_off_the_menu_raise(answer):
    with pytest.raises(ValueError, match="choice"):
        parse_choice(answer, 9)


NOT_JSON = "<html>not a model</html>"
"""A reply the fake server sends as it is, not as JSON: another program on the port."""
CUT_SHORT = '{"status": "o'
"""A reply the fake server cuts short of the length it announced."""


class FakeServer:
    """llama-server's two endpoints on a local port: it records requests and replies as told."""

    def __init__(self):
        self.requests, self.status, self.reply = [], 200, None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                fake.requests.append(("GET", self.path, None))
                raw = fake.reply in (NOT_JSON, CUT_SHORT)
                self._send(fake.status, fake.reply if raw else {"status": "ok"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(("POST", self.path, body))
                self._send(fake.status, fake.reply)

            def _send(self, status, payload):
                raw = payload in (NOT_JSON, CUT_SHORT)
                data = payload.encode() if raw else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data) + 10 * (payload == CUT_SHORT)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_server():
    server = FakeServer()
    yield server
    server.close()


def test_llama_server_sends_an_openai_request_with_the_schema(fake_server):
    fake_server.reply = {"choices": [{"message": {"content": '{"choice": 2}'}}]}
    chat = LlamaServer(url=fake_server.url, temperature=0.5, max_tokens=50)
    messages: list[Message] = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "go"},
    ]
    schema = {"type": "object"}
    assert chat(messages, schema, 17) == '{"choice": 2}'
    [(method, path, body)] = fake_server.requests
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert body["messages"] == messages
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": schema},
    }
    assert (body["seed"], body["temperature"], body["max_tokens"]) == (17, 0.5, 50)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_llama_server_errors_become_connection_errors(fake_server):
    chat = LlamaServer(url=fake_server.url)
    fake_server.status, fake_server.reply = 500, {"error": "boom"}
    with pytest.raises(ConnectionError, match="answered 500"):
        chat([], {}, 0)
    assert not chat.ready()
    fake_server.status, fake_server.reply = 200, {"choices": []}
    with pytest.raises(ConnectionError, match="unexpected reply"):
        chat([], {}, 0)
    assert chat.ready()


def test_a_reply_that_is_not_json_is_a_connection_error(fake_server):
    chat = LlamaServer(url=fake_server.url)
    fake_server.reply = NOT_JSON
    with pytest.raises(ConnectionError, match="not JSON"):
        chat([], {}, 0)
    assert not chat.ready()


def test_a_reply_cut_short_is_a_connection_error(fake_server):
    chat = LlamaServer(url=fake_server.url)
    fake_server.reply = CUT_SHORT
    with pytest.raises(ConnectionError, match="not JSON"):
        chat([], {}, 0)
    assert not chat.ready()


def test_no_server_says_how_to_start_one():
    with ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler) as taken:
        port = taken.server_address[1]
    chat = LlamaServer(url=f"http://127.0.0.1:{port}", timeout=5.0)
    with pytest.raises(ConnectionError, match="docker compose up"):
        chat([], {}, 0)
    assert not chat.ready()
