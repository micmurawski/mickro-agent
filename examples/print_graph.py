"""Print a loop as Mermaid or DOT, optionally render it to a local PNG/SVG."""

import argparse

from mickro_agent import PlanThenAct, ReAct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", choices=["react", "plan"], default="react")
    parser.add_argument("--format", choices=["mermaid", "dot"], default="mermaid")
    parser.add_argument("--direction", choices=["LR", "RL", "TB", "BT"], default="LR")
    parser.add_argument("--output", help="Optional .png or .svg file (requires Graphviz)")
    args = parser.parse_args()
    flow = (ReAct() if args.loop == "react" else PlanThenAct()).to_flow()
    export = flow.to_mermaid if args.format == "mermaid" else flow.to_dot
    print(export(direction=args.direction), end="")
    if args.output:
        flow.render(args.output, direction=args.direction)


if __name__ == "__main__":
    main()
