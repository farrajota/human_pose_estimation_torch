import optparse

def main():
	parser = optparse.OptionParser()

	parser.add_option('-q', '--query',
	    action="store", dest="query",
	    help="query string", default="spam")

	options, args = parser.parse_args()

	print 'Query string:', options.query

if __name__ == "__main__":
    main()
