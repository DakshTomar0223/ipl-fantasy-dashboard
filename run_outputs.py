"""Command-line entry point for the same JSON report used by the UI."""
import argparse
import json
from src.ui_backend import FantasyService


def main():
    parser = argparse.ArgumentParser(description="Generate historical IPL fantasy reports.")
    parser.add_argument("--date", help="DD-MM-YYYY; nearest available date is used by default")
    parser.add_argument("--match-id", help="Choose a specific match, including on doubleheader days")
    parser.add_argument("--exact-date", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Build historical features and fit the cutoff models")
    parser.add_argument("--evaluate", action="store_true", help="Save chronological held-out metrics")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()
    if not any([args.date, args.match_id, args.prepare, args.evaluate]):
        parser.error("Choose --date DD-MM-YYYY, --match-id ID, --prepare or --evaluate")
    print("Loading Excel and earlier-match features...", flush=True)
    service = FantasyService()
    if args.prepare:
        print(json.dumps(service.prepare(), indent=2))
    if args.evaluate:
        result = service.evaluate()
        path = service.root / "outputs/evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"Saved: {path}")
    if args.date or args.match_id:
        report = service.get_match_report(match_id=args.match_id, date=args.date,
                    nearest=not args.exact_date, charts=not args.no_charts)
        if report["status"] == "choose_match":
            if not report["matches"]:
                print("No match on that exact date. Omit --exact-date to find the nearest date.")
            for match in report["matches"]:
                print(f"{match['date']}: {' vs '.join(match['teams'])} | --match-id {match['match_id']}")
            return
        print(f"Match {report['match']['match_id']} | {report['match']['date']} | {report['status']}")
        print(json.dumps(report["comparison"], indent=2))
        print(json.dumps(report["team_win_probabilities"]["probabilities"], indent=2))
        for message in report["selection_errors"]:
            print(f"Selection issue: {message}")
        print(f"Saved: {service.root / 'outputs/matches' / report['match']['match_id']}")


if __name__ == "__main__":
    main()
