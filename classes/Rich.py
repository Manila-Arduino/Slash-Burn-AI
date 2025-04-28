from rich.console import Console, JustifyMethod
from rich.theme import Theme
from dataclasses import dataclass
from typing import List, Literal, Optional, Callable
from rich.prompt import Prompt
from rich.text import Text
from InquirerPy.resolver import prompt

custom_theme = Theme(
    {
        "info": "medium_turquoise",
        "warning": "yellow",
        "danger": "red",
        "success": "bold green",
    }
)

console = Console(theme=custom_theme)


def custom_ask(prompt: str) -> str:
    styled_prompt = Text(f"{prompt}: ", style="chartreuse1")
    console.print(styled_prompt, end="")
    # Taking input without trimming spaces
    user_input = input()
    return user_input


@dataclass
class Rich:
    @staticmethod
    def print(
        text: str,
        style: Optional[Literal["info", "warning", "danger", "success"]] = None,
        justify: Optional[JustifyMethod] = None,
        end: str = "\n",
    ):
        console.print(text, style=style, justify=justify, end=end)

    @staticmethod
    def welcome(title: str):
        Rich.print(
            f"---  :star2: {title} :star2:  ---",
            style="warning",
            justify="center",
        )

    @staticmethod
    def success():
        Rich.print(
            "---   :star2:  Success  :star2:   ---",
            style="success",
            justify="center",
        )

    @staticmethod
    def error(e: Exception | str):
        Rich.print(
            f"---   Error: {e}   ---",
            style="danger",
            justify="center",
        )

    @staticmethod
    def warning(text: str):
        Rich.print(
            text,
            style="warning",
            justify="center",
        )

    @staticmethod
    def wrap_init(title: str):
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                try:
                    Rich.welcome(title)
                    x = func(*args, **kwargs)
                    Rich.success()
                except Exception as e:
                    Rich.error(e)
                    raise e

            return wrapper

        return decorator

    @staticmethod
    def ask(text: str):
        return Prompt.ask(f"\n[chartreuse1]{text}[/chartreuse1]")

    @staticmethod
    def ask_yn(text: str):
        _input = Prompt.ask(f"\n[chartreuse1]{text}  (y/n)[/chartreuse1]")
        return _input in ["y", "Y", "Yes"]

    @staticmethod
    def ask_multiline(text: str, end: str):
        print()
        lines: List[str] = []
        while True:
            line = custom_ask(text)
            lines.append(line)
            # print(lines)
            if line == end:
                break

        return "\n".join(lines)

    @staticmethod
    def ask_choices(text: str, choices: List[str]) -> str:
        questions = [
            {
                "type": "list",
                "name": "choice",
                "message": f"{text}: ",
                "choices": choices,
            }
        ]

        answers = prompt(
            questions,
            style={
                "questionmark": "#ff9d00 bold",
                "question": "#87ff00",
                "pointer": "#ff7b00",
                "answered_question": "#87ff00",
                "answer": "#ff7b00",
            },
        )

        return answers["choice"]

    @staticmethod
    def info(
        text: str,
        emoji: str = ":rocket:",
        style: Literal["info", "warning", "danger", "success"] = "info",
    ):
        Rich.print(f"{emoji} {text}", style=style)

    @staticmethod
    def wrap(text: str):
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                Rich.print(f":rocket: {text}", style="info")
                # Rich.print(f":hot_beverage: {text}", style="info")
                return func(*args, **kwargs)

            return wrapper

        return decorator
