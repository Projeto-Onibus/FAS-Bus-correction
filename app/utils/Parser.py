import argparse
import pathlib
import datetime

def parse_args():
	parser = argparse.ArgumentParser(description="Trajectory classifier with GPU acceleration. Classifies long bus paths by line.")
# Common options
	common_parser = parser.add_argument_group(title="Common",
	description="Defines common options"
	)

	common_parser.add_argument('-c','--config',
	default=None,
	type=pathlib.Path,
	help="New configuration path. Overrides default path."
	)

	common_parser.add_argument("-v",'--verbose',
	action="count",
	default=0,
	help="Increase output verbosity"
	)


# Program parameters 
	data_parser = parser.add_argument_group(title="Data",
	description="Options for setting data settings such as date")
	data_parser.add_argument("-d","--date",
	type=datetime.date.fromisoformat,
	default=None,
	required=True,
	help="(YYYY-MM-DD) Date to query bus paths"
	)


# Execution parameters
# TODO: Control for max memory occupation size


# Debugging parameters
	debug_parser = parser.add_argument_group(title="Debug:")
	parser.add_argument("--status",
	action="store_true",
	help="Checks integrity of database connection and exits"
	)

# Filtering options 
	filtering_parser = parser.add_argument_group(title="Filtering inputs",
	description=""
	)
	filtering_parser.add_argument("-bb","--blacklist-buses",
	action="store_true",
	help="Enables blacklisting for buses. List read from files."
	)
	filtering_parser.add_argument("-bl","--blacklist-lines",
	action="store_true",
	help="Enables blacklisting for lines. List read from files."
	)
	filtering_parser.add_argument("-wb","--whitelist-buses",
	action="store_true",
	help="Enable whitelisting of bus inputs. List read from files."
	)
	filtering_parser.add_argument("-wl","--whitelist-lines",
	action="store_true",
	help="Enable whitelisting of lines inputs. List read from files."
	)
	filtering_parser.add_argument("-e","--everything",
	action="store_true",
	help="Flag to indicate that program can run without blacklist or whitelist enabled. Cannot be used with either -b or -w"
	)
	filtering_parser.add_argument("--bus-whitelist-path",
	type=pathlib.Path,
	default=pathlib.Path("/filters/bus_whitelist"),
	help="Path and filename to bus identifier's whitelist. Either comma or line separated. Only those IDs will be parsed."
	)
	filtering_parser.add_argument('--line-whitelist-path',
	type=pathlib.Path,
	default=pathlib.Path("/filters/lines_whitelist"),
	help="Path and filename to line identifier's whitelist. Either comma or line separated. Only those IDs will be parsed."
	)
	filtering_parser.add_argument("--bus-blacklist-path",
	type=pathlib.Path,
	default=pathlib.Path("/filters/bus_blacklist"),
	help="Path and filename to bus identifier's whitelist. Either comma or line separated. Those IDs will be excluded from list."
	)
	filtering_parser.add_argument("--line-blacklist-path",
	type=pathlib.Path,
	default=pathlib.Path("/filters/lines_blacklist"),
	help="Path and filename to line identifier's whitelist. Either comma or line separated. Those IDs will be excluded from list."
	)
	return parser.parse_args()
