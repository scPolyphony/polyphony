"""Console script for polyphony."""
import argparse
import sys

from polyphony.router.app import create_app


def main():
    """Console script for polyphony."""
    parser = argparse.ArgumentParser()

    # API flag
    parser.add_argument('--host', default='127.0.0.1', help='The host to run the server')
    parser.add_argument('--port', default=7777, help='The port to run the server')
    parser.add_argument('--debug', action="store_true", help='Run Flask in debug mode')
    parser.add_argument('--problem', default='pancreas_easy', help='The identifier of the problem')

    args = parser.parse_args()
    server_app = create_app(args.problem)

    server_app.run(
        debug=args.debug,
        host=args.host,
        port=args.port
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
