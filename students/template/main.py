from pathlib import Path
import runpy


def main():
    demo_entry = Path(__file__).parent / "src" / "week06-demo" / "main.py"
    runpy.run_path(str(demo_entry), run_name="__main__")


if __name__ == "__main__":
    main()
