from rich import print


class Rich:
    @staticmethod
    def print(text):
        print(text)

    @staticmethod
    def printr(text):
        print(f"[red]{text}[/red]")

    @staticmethod
    def printg(text):
        print(f"[green]{text}[/green]")

    @staticmethod
    def printb(text):
        print(f"[blue]{text}[/blue]")

    @staticmethod
    def printy(text):
        print(f"[yellow]{text}[/yellow]")

    @staticmethod
    def printo(text):
        print(f"[orange]{text}[/orange]")

    @staticmethod
    def printm(text):
        print(f"[magenta]{text}[/magenta]")
