"""Players that ask a language model: a numbered menu of actions, and an answer that picks one.

Each decision becomes one chat request. A game's `Menu` describes the legal actions as
options, in words, and writes the chat: a system message that stays the same all game (the
rules), so a server can reuse its work from one request to the next, and a user message with
what the player sees and the options. The answer must fit a JSON schema whose ``choice`` is one
of the menu's numbers, so a server that enforces schemas (llama.cpp does) can only answer with
a legal action. `LLMPlayer` plays any game with a menu.

The model is reached through `Chat`, any callable that takes the messages and the schema and
returns the answer's text. `LlamaServer` talks to llama.cpp's ``llama-server``, which
``compose.yaml`` at the repository's root starts, over HTTP with the standard library. Tests
pass a fake `Chat`.
"""

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from gridzero.core._arrays import check_int, check_real
from gridzero.game import Action

type Schema = Mapping[str, object]
"""A JSON schema, as Python values."""


class Message(TypedDict):
    """One message of a chat."""

    role: Literal["system", "user", "assistant"]
    content: str


class Chat(Protocol):
    """A language model: answers a chat with text that fits a JSON schema."""

    def __call__(self, messages: Sequence[Message], schema: Schema, seed: int) -> str:
        """Return the model's answer.

        Parameters
        ----------
        messages : sequence of Message
            The chat so far: a system message, then a user message.
        schema : mapping
            The JSON schema the answer must fit.
        seed : int
            The seed of the model's sampling, non-negative, so that a game can replay.

        Returns
        -------
        str
            The answer: a JSON document that fits ``schema``, if the model kept to it.
        """
        ...


@dataclass(frozen=True)
class LlamaServer:
    """A chat with llama.cpp's ``llama-server``, through its OpenAI-style HTTP API.

    The defaults suit Qwen3.5 without thinking, as its model card recommends them. The server's
    own model is used: ``llama-server`` serves one.

    Parameters
    ----------
    url : str, default "http://127.0.0.1:8080"
        Where the server listens, as ``compose.yaml`` starts it.
    temperature, top_p : float, default 0.7 and 0.8
        Sampling settings, non-negative.
    top_k : int, default 20
        Sample among the ``top_k`` likeliest tokens, at least 1.
    max_tokens : int, default 400
        The longest answer, at least 1.
    thinking : bool, default False
        Let the model think before it answers, for models whose chat template has the switch
        (``enable_thinking``). Thinking makes each decision several times slower.
    timeout : float, default 300.0
        Seconds to wait for an answer, positive.

    Raises
    ------
    TypeError, ValueError
        If a setting is of the wrong type or out of range.
    """

    url: str = "http://127.0.0.1:8080"
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 400
    thinking: bool = False
    timeout: float = 300.0

    def __post_init__(self) -> None:
        check_real(self.temperature, "temperature", 0.0)
        check_real(self.top_p, "top_p", 0.0, 1.0)
        check_int(self.top_k, "top_k", 1)
        check_int(self.max_tokens, "max_tokens", 1)
        check_real(self.timeout, "timeout", 0.0)
        if self.timeout == 0:
            raise ValueError("timeout must be positive")
        if not isinstance(self.thinking, bool):
            raise TypeError(f"thinking must be a bool, not {type(self.thinking).__name__}")

    def __call__(self, messages: Sequence[Message], schema: Schema, seed: int) -> str:
        """Return the model's answer (see `Chat`).

        Raises
        ------
        ConnectionError
            If the server can't be reached, answers with an error, or answers with something
            other than a chat completion in JSON.
        """
        body = {
            "messages": list(messages),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        reply = self._request("/v1/chat/completions", body)
        choices = reply.get("choices") if isinstance(reply, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ConnectionError(f"unexpected reply from {self.url}: {reply!r}")
        return content

    def ready(self) -> bool:
        """Return whether the server is up with its model loaded.

        Returns
        -------
        bool
            True when ``/health`` answers OK; False while the model loads or if nothing
            listens.
        """
        try:
            self._request("/health", None)
        except ConnectionError:
            return False
        return True

    def _request(self, path: str, body: object) -> object:
        """Send a POST of ``body`` as JSON (a GET if it is None) and return the parsed reply."""
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            with error:  # it holds the reply's body open
                detail = error.read().decode(errors="replace")[:500]
            raise ConnectionError(f"{self.url}{path} answered {error.code}: {detail}") from error
        except (json.JSONDecodeError, http.client.HTTPException) as error:
            # Another program on the port, or a reply cut short (IncompleteRead)
            raise ConnectionError(f"{self.url}{path} answered, but not JSON: {error}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ConnectionError(
                f"no LLM server at {self.url} ({error}); start one with `docker compose up -d`"
            ) from error


class Option(NamedTuple):
    """One entry of a menu: an action and what it would do, in words."""

    action: Action
    text: str


class Menu[G, V](Protocol):
    """What a game tells a language model: its legal actions as options, and the chat."""

    def options(self, game: G, view: V, legal: NDArray[np.bool_], /) -> list[Option]:
        """Return the legal actions as a menu, each described in words.

        Parameters
        ----------
        game : G
            The rules.
        view : V
            What the acting player sees (`game.Game.view`).
        legal : ndarray of bool, shape (n_verbs, n_cells)
            The legal actions.

        Returns
        -------
        list of Option
            At least one option, each a legal action.
        """
        ...

    def messages(self, game: G, view: V, options: Sequence[Option], /) -> list[Message]:
        """Return the chat of one decision: the rules, then what the player sees and the menu.

        Parameters
        ----------
        game, view
            As for `options`.
        options : sequence of Option
            The menu, as `options` made it, to be numbered from 0.

        Returns
        -------
        list of Message
            The chat to send.
        """
        ...


@dataclass(frozen=True)
class LLMPlayer[G, V]:
    """Asks a language model to pick from a game's menu of the legal actions.

    A decision with a single option (often the end of a turn) is taken without asking. Each
    request's sampling seed comes from the player's generator, so a game replays when the
    model's server does.

    Parameters
    ----------
    menu : Menu
        The game's menu and chat.
    chat : Chat, default LlamaServer()
        The model.
    reason_chars : int, default 120
        The longest reasoning the model writes before its choice, in characters; 0 for none.
        Writing first may help a small model choose, at the cost of the tokens: 300
        characters took Qwen3.5-4B 1.5 s a decision in Tactics, against 0.66 s without.

    Raises
    ------
    TypeError, ValueError
        If ``reason_chars`` is not a non-negative integer.
    """

    menu: Menu[G, V]
    chat: Chat = LlamaServer()
    reason_chars: int = 120

    def __post_init__(self) -> None:
        check_int(self.reason_chars, "reason_chars", 0)

    def __call__(
        self, game: G, player: int, view: V, legal: NDArray[np.bool_], rng: np.random.Generator
    ) -> Action:
        """Return the model's choice (see `game.Player`).

        Raises
        ------
        ValueError
            If the model's answer is not JSON with a ``choice`` from the menu.
        ConnectionError
            If ``chat`` can't reach the model.
        """
        options = self.menu.options(game, view, legal)
        if len(options) == 1:
            return options[0].action
        messages = self.menu.messages(game, view, options)
        answer = self.chat(messages, self.schema(len(options)), int(rng.integers(2**31)))
        return options[parse_choice(answer, len(options))].action

    def schema(self, n_options: int) -> dict[str, object]:
        """Return the JSON schema of an answer.

        Parameters
        ----------
        n_options : int
            How many options the menu has.

        Returns
        -------
        dict
            An object with ``choice``, one of ``0 .. n_options - 1``, after ``reason``, a
            string of at most `reason_chars` characters, unless that is 0.
        """
        properties: dict[str, object] = {}
        if self.reason_chars:
            properties["reason"] = {"type": "string", "maxLength": self.reason_chars}
        properties["choice"] = {"type": "integer", "enum": list(range(n_options))}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }


def parse_choice(answer: str, n_options: int) -> int:
    """Return the option an answer picks.

    Parameters
    ----------
    answer : str
        The model's answer: a JSON object with an integer ``choice``.
    n_options : int
        How many options the menu has.

    Returns
    -------
    int
        The choice, in ``0 .. n_options - 1``.

    Raises
    ------
    ValueError
        If the answer is not such an object, or its choice is not a number on the menu.
    """
    try:
        choice = json.loads(answer)["choice"]
    except (json.JSONDecodeError, TypeError, KeyError) as error:
        raise ValueError(f"not an answer with a choice: {answer!r}") from error
    if isinstance(choice, bool) or not isinstance(choice, int) or not 0 <= choice < n_options:
        raise ValueError(f"choice {choice!r} is not on the menu of {n_options}: {answer!r}")
    return choice
