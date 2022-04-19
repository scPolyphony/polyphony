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
    parser.add_argument('--warnings', action="store_true", help='Log warnings')

    parser.add_argument('--problem', default='pancreas', help='The identifier of the problem')
    parser.add_argument('--experiment', default='case-1', help='The identifier of the experiment')
    parser.add_argument('--load_exist', action="store_true", help='Load existing results')
    parser.add_argument('--save', action="store_true", help='Save the results in each step')
    parser.add_argument('--eval', action="store_true", help='Whether run the evaluation')
    parser.add_argument('--iter', type=int, default=None,
                        help='Load certain step from an existing experiment')

    args = parser.parse_args()
    server_app = create_app(args)

    server_app.run(
        debug=args.debug,
        host=args.host,
        port=args.port
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
